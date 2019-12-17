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

struct Embed; w; end

function Embed(charsetsize::Int, embedsize::Int)
    Embed(param(embedsize, charsetsize))
end

function (l::Embed)(x)
    l.w[:, x]
end

struct Linear; w; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize))
end

function (l::Linear)(x)
    l.w * x
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
Base.eltype(::Type{WordsData}) = NTuple{2}

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
        src_length = length(src_word)
        
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
    src_lengths = map(x -> length(x), bucket)
    max_length = max(src_lengths...)
    x = zeros(Int64, length(bucket), max_length + 2) # default d.batchmajor is false

    for (i, v) in enumerate(bucket)
        to_be_added = fill(src_eow, max_length - length(v) + 1)
        x[i,:] = [src_eow; v; to_be_added]
    end

    
    d.batchmajor && (x = x')
    return (x[:, 1:end-1], x[:, 2:end])
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

# Utility to convert int arrays to sentence strings
function int2word(y, charset)
    y = vec(y)
    ysos = findnext(w->!isequal(w, charset.eow), y, 1)
    ysos === nothing && return ""
    yeos = something(findnext(isequal(charset.eow), y, ysos), 1+length(y))
    join(charset.i2c[y[ysos:yeos-1]], " ")
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

# RNN Language model
struct LModel
    srcembed::Embed
    rnn::RNN        
    projection::Linear  
    dropout::Real
    srccharset::Charset 
end

function LModel(hidden::Int, srcembsz::Int, srccharset::Charset;
             layers=1, dropout=0)
    
    srcembed = Embed(length(srccharset.i2c), srcembsz)
    rnn = RNN(srcembsz, hidden; bidirectional=false, numLayers=layers, dropout=dropout)
    projection = Linear(hidden, length(srccharset.i2c))
    
    LModel(srcembed, rnn, projection, dropout, srccharset)
end

# Language loss function
function (s::LModel)(src, tgt; average=true)
    s.rnn.h, s.rnn.c = 0, 0
    srcembed = s.srcembed(src)
    rnn_out = s.rnn(srcembed)
    dims = size(rnn_out)
    output = s.projection(dropout(reshape(rnn_out, dims[1], dims[2] * dims[3]), s.dropout))
    scores = reshape(output, size(output, 1), dims[2], dims[3])
    nll(scores, mask(tgt, s.srccharset.eow); dims=1, average=average)
end

# Generating words using the LM with sampling
function generate(s::LModel; start="", maxlength=30)
    s.rnn.h, s.rnn.c = 0, 0
    chars = fill(s.srccharset.eow, 1)
    
    starting_index = 1
    for i in 1:length(start)
        push!(chars, s.srccharset.c2i[start[i]])
        charembed = s.srcembed(chars[i:i])
        rnn_out = s.rnn(charembed)
        starting_index += 1
    end
    
    for i in starting_index:maxlength
        charembed = s.srcembed(chars[i:i])
        rnn_out = s.rnn(charembed)
        output = model.projection(dropout(rnn_out, model.dropout))
        push!(chars, s.srccharset.c2i[ sample(s.srccharset.i2c, Weights(Array(softmax(reshape(output, length(s.srccharset.i2c)))))) ] )
        
        if chars[end] == s.srccharset.eow
            break
        end
    end
    
    join([ s.srccharset.i2c[i] for i in chars ], "")
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

# Define Turkish char set
char_set = "ABCDEFGHIJKLMNOPRSTUVYZabcdefghijklmnoprstuvyzÇÖÜçöüĞğİıŞş"
datadir = "turkish_word_set"
BATCHSIZE, MAXLENGTH = 32, 25
println("Reading data from directory :", datadir)
println("Setting batch size to ",BATCHSIZE, " and max word length to ", MAXLENGTH)
tr_charset = Charset(char_set)
tr_train = TextReader("$datadir/train.tr", tr_charset)
tr_dev = TextReader("$datadir/dev.tr", tr_charset)
dtrn = WordsData(tr_train, batchsize=BATCHSIZE, maxlength=MAXLENGTH, bucketwidth = 3)
ddev = WordsData(tr_dev, batchsize=BATCHSIZE, maxlength=MAXLENGTH, bucketwidth = 3)

# Read words to check the generated words
println("Reading words sets for testing generated words...")
training_set = readwordset("$datadir/train.tr")
test_set = [ readwordset("$datadir/test.tr"); readwordset("$datadir/dev.tr") ]
println(length(training_set), " words in training set")
println(length(test_set), " words in test set")

@info "Training LModel"
epochs, model_size, layers = 4, 128, 2
println("epochs :", epochs)
println("model_size :", model_size)
println("layers :", layers)

println("Collecting training data")
ctrn = collect(ddev)
trnx10 = collect(flatten(shuffle!(ctrn) for i in 1:epochs))
trnmini = ctrn[1:20]
dev = collect(dtrn)

model = LModel(model_size, model_size, tr_charset; layers=layers, dropout=0.2)
model = train!(model, trnx10, dev, trnmini)

# Test generated words
println("Generating 100 words for test..")
generated_words = [ generate(model) for c in 1:100 ]
intest = [ w for w in generated_words if w in test_set]         
notintraining = [ w for w in generated_words if !(w in training_set)]

println(100 - length(notintraining) , "% of the generated words are words in training set")  
println(length(intest), "% of the generated words are real words that are in the test set")
println("\nExamples of the new generated words:")
println(join(notintraining, "\n"))

# Generate words that starts with some string = ge
start="ge"
generated_words = [ generate(model; start=start) for c in 1:20 ]
println("Generating words that starts with \"", start, "\" :")
println(join(notintraining, "\n"))