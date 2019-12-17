# Turkish word discriminator
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, CuArrays, Random, IterTools, StatsBase

struct Charset
    c2i::Dict{Any,Int}
    i2c::Vector{Any}
    eow::Int
end

function Charset(charset::String; eow="")
    i2c = [ eow; [ c for c in charset ]  ]
    c2i = Dict( c => i for (i, c) in enumerate(i2c))
    return Charset(c2i, i2c, c2i[eow])
end

struct TextReader
    file::String
    charset::Charset
end

function Base.iterate(r::TextReader, s=nothing)
    s === nothing && (s = open(r.file))
    eof(s) && return close(s)
    word, label = split(readline(s))
    return (([ get(r.charset.c2i, c, r.charset.eow) for c in word ], parse(Int, label) + 1), s)
end

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

struct WordsData
    src::TextReader        
    batchsize::Int         
    maxlength::Int         
    batchmajor::Bool       
    bucketwidth::Int    
    buckets::Vector        
    batchmaker::Function   
end

function WordsData(src::TextReader; batchmaker = arraybatch, batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 2, numbuckets = min(128, maxlength ÷ bucketwidth))
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    WordsData(src, batchsize, maxlength, batchmajor, bucketwidth, buckets, batchmaker)
end

Base.IteratorSize(::Type{WordsData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{WordsData}) = Base.HasEltype()
Base.eltype(::Type{WordsData}) = Tuple{Array{Int64,2},Array{Int64,1}}

function Base.iterate(d::WordsData, state=nothing)
    if state == 0 # When file is finished but buckets are partially full 
        for i in 1:length(d.buckets)
            if length(d.buckets[i]) > 0
                batch = d.batchmaker(d, d.buckets[i])
                d.buckets[i] = []
                return batch, state
            end
        end
        return nothing # Finish iteration
    elseif state === nothing
        # Just to make sure
        for i in 1:length(d.buckets)
            d.buckets[i] = []
        end
        state = nothing
    end

    while true
        src_next = iterate(d.src, state)
        
        if src_next === nothing
            state = 0
            return iterate(d, state)
        end
        
        (src_word, src_state) = src_next
        state = src_state
        src_length = length(src_word[1])
        
        (src_length > d.maxlength) && continue

        i = Int(ceil(src_length / d.bucketwidth))
        i > length(d.buckets) && (i = length(d.buckets))

        push!(d.buckets[i], src_word)
        if length(d.buckets[i]) == d.batchsize
            batch = d.batchmaker(d, d.buckets[i])
            d.buckets[i] = []
            return batch, state
        end
    end
end

function arraybatch(d::WordsData, bucket)
    src_eow = d.src.charset.eow
    
    x = zeros(Int64, length(bucket), d.maxlength) # default d.batchmajor is false
    for (i, v) in enumerate(bucket)
        to_be_added = fill(src_eow, d.maxlength - length(v[1]))
        x[i,:] = [v[1]; to_be_added]
    end
    
    y = [ x[2] for x in bucket]
    
    d.batchmajor && (x = x')
    return (x, y)
end


struct Embed; w; end
Embed(charsetsize::Int, embedsize::Int) = Embed(param(embedsize, charsetsize))
(l::Embed)(x) = (em=permutedims(l.w[:, x], [3, 1, 2]); ds=size(em); em=reshape(em, ds[1], ds[2], 1, ds[3])) # (E, B, T) -> (T, E, 1, B)

struct Conv; w; b; f; p; end
(c::Conv)(x) = (co=conv4(c.w, dropout(x,c.p)); c.f.(pool((co .+ c.b); window=(size(x, 1), size(x, 2)))))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

# Perform convolution then, global-max pooling and concatenate the output and feed it to sequential dense layer 
mutable struct DisModel
    charset::Charset
    embed::Embed
    filters
    dense_layers
end

function DisModel(charset, embeddingsize, filters, denselayers)
    Em = Embed(length(charset.i2c), embeddingsize)
    Em.w[:, charset.eow] = KnetArray(zeros(embeddingsize))
    DisModel(charset, Em, filters, denselayers)
end

function (c::DisModel)(x)
    em = c.embed(x)
    filters_out = []
    for f in c.filters
        push!(filters_out, f(em))
    end
    out = cat(filters_out...;dims=3)
    for l in c.dense_layers
        out = l(out)
    end
    out
end

(c::DisModel)(x,y; average=true) = nll(c(x), y; average=average)

# per-word loss (in this case per-batch loss)
function loss(model, data; average=true)
    l = 0
    n = 0
    a = 0
    for (x, y) in data
        v = model(x, y; average=false)
        l += v[1]
        n += v[2]
        a += (v[1] / v[2])
    end
    average && return a
    return l, n
end


function train!(model, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), seconds=30) do y
        devloss = loss(model, dev)
        tstloss = map(d->loss(model,d), tst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (dev=devloss, tst=tstloss, mem=Float32(CuArrays.usage[]))
    end
    return bestmodel
end