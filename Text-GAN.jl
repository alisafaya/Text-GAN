using Knet, Test, Base.Iterators, Printf, LinearAlgebra, CuArrays, Random, IterTools, StatsBase

struct Charset
    c2i::Dict{Any,Int}
    i2c::Vector{Any}
    eow::Int
end

function Charset(charset::String; eow="")
    i2c = [ eow; [ c for c in charset ]  ]
    print(i2c)
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
    return [ get(r.charset.c2i, c, r.charset.eow) for c in readline(s)], s
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
end

function WordsData(src::TextReader; batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 2, numbuckets = min(128, maxlength ÷ bucketwidth))
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    WordsData(src, batchsize, maxlength, batchmajor, bucketwidth, buckets)
end

Base.IteratorSize(::Type{WordsData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{WordsData}) = Base.HasEltype()
Base.eltype(::Type{WordsData}) = Array{Any,1}

function Base.iterate(d::WordsData, state=nothing)
    if state == 0 # When file is finished but buckets are partially full 
        for i in 1:length(d.buckets)
            if length(d.buckets[i]) > 0
                buc = d.buckets[i]
                d.buckets[i] = []
                return buc, state
            end
        end
        return nothing # Finish iteration
    end

    while true
        src_next = iterate(d.src, state)
        
        if src_next === nothing
            state = 0
            return iterate(d, state)
        end
        
        (src_word, src_state) = src_next
        state = src_state
        src_length = length(src_word)
        
        (src_length > d.maxlength) && continue

        i = Int(ceil(src_length / d.bucketwidth))
        i > length(d.buckets) && (i = length(d.buckets))

        push!(d.buckets[i], src_word)
        if length(d.buckets[i]) == d.batchsize
            buc = d.buckets[i]
            d.buckets[i] = []
            return buc, state
        end
    end
end

function arraybatch(d::WordsData, bucket)
    src_eow = d.src.charset.eow
    src_lengths = map(x -> length(x), bucket)
    max_length = max(src_lengths...)
    x = zeros(Int64, length(bucket), d.maxlength + 1) # default d.batchmajor is false

    for (i, v) in enumerate(bucket)
        to_be_added = fill(src_eow, d.maxlength - length(v))
        x[i,:] = [src_eow; v; to_be_added]
    end
    
    d.batchmajor && (x = x')
    return (x[:, 1:end-1], x[:, 2:end]) # to calculate nll on generators output directly
end

function readwordset(fname)
    words = []
    fi = open(fname)
    while !eof(fi)
        push!(words, readline(fi))
    end
    close(fi)
    words
end

function mask(a, pad)
    a = copy(a)
    for i in 1:size(a, 1)
        j = size(a,2)
        while a[i, j] == pad && j > 1
            if a[i, j - 1] == pad
                a[i, j] = 0
            end
            j -= 1
        end
    end
    return a
end

struct Embed; w; end

function Embed(shape...)
    Embed(param(shape...))
end

get_z(shape...) = KnetArray(randn(Float32, shape...))

# this function is similar to gumble softmax, it is used to soften the one-hot-vector of the real samples
# tau -> normalization factor; the bigger the softer
function soften(A; dims=1, tau=0.5, norm_factor=0.01) 
    A = (A .+ norm_factor) ./ tau
    softmax(A; dims=dims)
end

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

function (l::Embed)(x)
    dims = size(x)
    em = l.w * reshape(x, dims[1], dims[2] * dims[3]) # reshape for multiplication 
    em = reshape(em, size(em, 1), dims[2], dims[3]) # reshape to original size
end

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 3-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

mutable struct DisModel
    charset::Charset
    embed::Embed
    rnn::RNN
    denselayers
end

# This discriminator uses separate weights for its embedding layer
function DisModel(charset, embeddingSize::Int, hidden, denselayers; layers=1, dropout=0)
    Em = Embed(embeddingSize, length(charset.c2i))
    rnn = RNN(embeddingSize, hidden; numLayers=layers, dropout=dropout)
    DisModel(charset, Em, rnn, denselayers)
end

function (c::DisModel)(x) # the input here is weights of the characters with shape (C, B, T)
    c.rnn.h, c.rnn.c = 0, 0
    em = c.embed(x)
    rnn_out = c.rnn(em)
    dims = size(rnn_out)
    rnn_out = reshape(rnn_out, dims[1], dims[2] * dims[3] )
    for l in c.denselayers
        rnn_out = l(rnn_out)
    end
    reshape(rnn_out, :, dims[2], dims[3])
end

function (c::DisModel)(x, reward::Int; average=true)
    scores = softmax(c(x))
    scores = reshape(scores, :, size(scores, 2) * size(scores, 3))
    -log.(scores[1, :])
end

function (c::DisModel)(x, y; average=true)
    scores = reshape(c(x), :, size(y, 1) * size(y, 2))
    labels = reshape(y, size(y, 1) * size(y, 2))
    return nll(scores, y; average=average)
end

# concatenate z with embedding vectors, z -> (z_size, B), returns (E+z_size, B, T)
# this will be used to feed Z to generator at each timestep
function (l::Embed)(x, z)
    em = l.w[:, x]
    z_array = cat((z for i in 1:size(em, 3))...; dims=(3))
    cat(em, z_array; dims=(1))
end

# Generator model
struct GenModel
    embed::Embed
    rnn::RNN        
    dropout::Real
    charset::Charset
    projection::Embed
    disModel::DisModel
    maxlength::Int
    zsize::Int
end

function GenModel(esize::Int, zsize::Int, hidden::Int, charset::Charset, disModel::DisModel, maxlength::Int; layers=2, dropout=0)
    embed = Embed(esize, length(charset.i2c))
    rnn = RNN(zsize + esize, hidden; numLayers=layers, dropout=dropout)
    projection = Embed(hidden, length(charset.i2c))
    GenModel(embed, rnn, dropout, charset, projection, disModel, maxlength, zsize)
end

# This generator shares the projection layers weights of the discriminator for its projection layer
function GenModel(esize::Int, zsize::Int, charset::Charset, disModel::DisModel, maxlength::Int; layers=2, dropout=0)
    embed = Embed(esize, length(charset.i2c))
    rnn = RNN(zsize + esize, size(disModel.embed.w, 1); numLayers=layers, dropout=dropout)
    GenModel(embed, rnn, dropout, charset, disModel.embed, disModel, maxlength, zsize)
end

# Generator forward pass using Z and Teacher forcing for input
function (s::GenModel)(GenInput) # tuple (input, Z)
    (input, _), Z = GenInput
    s.rnn.h, s.rnn.c = 0, 0
    input = s.embed(input, Z)
    rnn_out = s.rnn(input)
    dims = size(rnn_out)
    output = s.projection.w' * reshape(rnn_out, dims[1], dims[2] * dims[3])
    scores = reshape(output, size(output, 1), dims[2], dims[3])
end

# Generator loss
function (s::GenModel)(GenInput, calculateloss::Int; average=true)
    # since the discriminator will output 2 for the fake data, 
    #    we train the generator to get 1 as output from the discriminator
    (_, output), Z = GenInput
    x = s(GenInput)
    dloss = s.disModel(softmax(x), 1)
    scores = reshape(x, :, size(output, 1) * size(output, 2))
    output = mask(reshape(output, size(output, 1) * size(output, 2)), s.charset.eow)
    glosses = [nll(scores[:, i], output[i:i]) * dloss[i] for i in 1:size(output, 1) ]
    average && return mean(glosses)
    return sum(glosses), length(glosses)
end

function generate(s::GenModel; start="", maxlength=30)
    s.rnn.h, s.rnn.c = 0, 0
    Z = get_z(s.zsize, 1, 1)
    chars = fill(s.charset.eow, 1)

    starting_index = 1
    for i in 1:length(start)
        push!(chars, s.charset.c2i[start[i]])
        charembed = s.embed(chars[i:i], Z)
        rnn_out = s.rnn(charembed)
        starting_index += 1
    end
    
    for i in starting_index:maxlength
        charembed = s.embed(chars[i:i], Z)
        rnn_out = s.rnn(charembed)
        dims = size(rnn_out)
        output = s.projection.w' * reshape(rnn_out, dims[1], dims[2] * dims[3])
        push!(chars, s.charset.c2i[ sample(s.charset.i2c, Weights(Array(softmax(reshape(output, length(s.charset.i2c)))))) ] )
#         push!(chars, argmax(output)[1])
        if chars[end] == s.charset.eow
            break
        end
    end
    
    join([ s.charset.i2c[i] for i in chars ], "")
end

struct Sampler
    wordsdata::WordsData
    charset::Charset
    genModel::GenModel
    maxBatchsize::Int
end

Base.IteratorSize(::Type{Sampler}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{Sampler}) = Base.HasEltype()
Base.eltype(::Type{Sampler}) = Tuple{KnetArray{Float32,3},Array{Int64,2}}

function Base.iterate(s::Sampler, state=nothing)
    wdatastate = iterate(s.wordsdata, state)
    wdatastate === nothing && (return nothing)
    
    (bucket, state) = wdatastate
    bsize = length(bucket)
    src_eow = s.charset.eow
    src_lengths = map(x -> length(x), bucket)
    max_length = max(src_lengths...)
    gsize = bsize
    generated = softmax(s.genModel((arraybatch(s.wordsdata, bucket), get_z(s.genModel.zsize, gsize, 1))))

    to_be_cat = [generated, ]
    for (i, v) in enumerate(bucket)
        tindex = [i for i in 1:length(v)]
        pindex = [i for i in length(v)+1:s.wordsdata.maxlength]
        onehot = KnetArray(zeros(Float32, length(s.charset.c2i), 1, s.wordsdata.maxlength))
        onehot[v, :, tindex] .= 1
        onehot[s.charset.eow, :, pindex] .= 1
        onehot = soften(onehot) # soften one hot vectors elements value
        push!(to_be_cat, onehot)
    end
    x = cat(to_be_cat...;dims=2) # concatenate both generated and sampled words

    y = Array(ones(Int, gsize+bsize, s.wordsdata.maxlength)) # create labels 1 -> real, 2-> not-real
    y[1:gsize, :] = y[1:gsize, :] .+ 1
    
    ind = shuffle(1:gsize+bsize) # used to shuffle the batch
    x, y = x[:, ind, :], y[ind, :]
    return (x,y), state
end

function train!(model, parameters, trn, dev, tst; lr=0.001)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn; lr=lr, params=parameters), seconds=30) do y
        devloss = loss(model, dev)
        tstloss = loss(model, tst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (dev=devloss, tst=tstloss, mem=Float32(CuArrays.usage[]))
    end
    return bestmodel
end

char_set = "ABCDEFGHIJKLMNOPRSTUVYZabcdefghijklmnoprstuvyzÇÖÜçöüĞğİıŞş"
tr_charset = Charset(char_set)
datadir = "turkish_word_set"
BATCHSIZE = 128
MAXLENGTH = 15
tr_dev = TextReader("$datadir/dev.tr", tr_charset)
tr_trn = TextReader("$datadir/train.tr", tr_charset)
dtrn = WordsData(tr_trn, batchsize=BATCHSIZE, maxlength=MAXLENGTH, bucketwidth = 1)
ddev = WordsData(tr_dev, batchsize=BATCHSIZE, maxlength=MAXLENGTH, bucketwidth = 1)

EMBEDDING_SIZE = 256
DHIDDEN_SIZE = 128
GDROPOUT = 0.1
DDROPOUT = 0.3

dismodel = DisModel(tr_charset, EMBEDDING_SIZE, DHIDDEN_SIZE,(
        Dense(DHIDDEN_SIZE, 2, identity),
        ); dropout=DDROPOUT)

GH_SIZE = 256
Z_SIZE = 128

genmodel = GenModel(EMBEDDING_SIZE, Z_SIZE, GH_SIZE, tr_charset, dismodel, MAXLENGTH; dropout=GDROPOUT, layers=2)
trnsampler = Sampler(dtrn, tr_charset, genmodel, BATCHSIZE * 2)
devsampler = Sampler(ddev, tr_charset, genmodel, BATCHSIZE * 2)

ctrn = collect(dtrn)
cdev = collect(ddev)
collecttrn = [ ((arraybatch(dtrn, i), get_z(Z_SIZE, size(i, 1), 1)), 1) for i in ctrn ]
collectdev = [ ((arraybatch(ddev, i), get_z(Z_SIZE, size(i, 1), 1)), 1) for i in cdev ]

function gmodel(batches)
    global genmodel
    global collecttrn
    global collectdev
    
    trnxbatches = shuffle!(collecttrn)[1:batches]
    devbatches = shuffle!(collectdev)
    trnmini = trnxbatches[1:5]

    genmodel = train!(genmodel, params(genmodel)[1:3], trnxbatches, devbatches, trnmini)
end

function dmodel(batches)
    global trnsampler
    global devsampler
    global dismodel
    
    ctrn = collect(trnsampler)
    ctrn = shuffle!(ctrn)[1:batches]
    trnmini = ctrn[1:5]
    dev = collect(devsampler)
    dismodel = train!(dismodel, params(dismodel), ctrn, dev, trnmini) 
end

@info "Started training..."
for k in 1:5
    println("Turn no:", k)
    println("Ex.Generated words: \n", join([ generate(genmodel; maxlength=MAXLENGTH) for i in 1:5 ],"\n"))
    dmodel(50)
    gmodel(400)
end

Knet.save("text-gan-model.jld2", "genmodel", genmodel)