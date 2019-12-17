include("discriminator.jl")

# Define Turkish char set
char_set = "ABCDEFGHIJKLMNOPRSTUVYZabcdefghijklmnoprstuvyzÇÖÜçöüĞğİıŞş"
datadir = "discriminator_labeled_set"
BATCHSIZE, MAXLENGTH = 32, 25
println("Reading data from directory:", datadir)
println("Setting batch size to ",BATCHSIZE, " and max word length to ", MAXLENGTH)
tr_charset = Charset(char_set)
tr_train = TextReader("$datadir/dis.train", tr_charset)
tr_dev = TextReader("$datadir/dis.dev", tr_charset)
dtrn = WordsData(tr_train, batchsize=BATCHSIZE, maxlength=MAXLENGTH, bucketwidth = 1)
ddev = WordsData(tr_dev, batchsize=BATCHSIZE, maxlength=MAXLENGTH, bucketwidth = 1)

@info "Initializing CNN Model"
embedding_size, filter_no = 64, 20
println("embeding_size: ", embedding_size)
println(filter_no, " Convolution filter for each region size (2, 3, 4)")

model = DisModel(tr_charset, embedding_size, (
        Conv(2,embedding_size,1,filter_no; pdrop=0.2),
        Conv(3,embedding_size,1,filter_no; pdrop=0.2),
        Conv(4,embedding_size,1,filter_no; pdrop=0.2),
        ),(
        Dense(60,64,pdrop=0.3),
        Dense(64,2,sigm,pdrop=0.3)
        ))

@info "Collecting training and dev data..."
epochs = 10
ctrn = collect(dtrn)
trnx10 = collect(flatten(shuffle!(ctrn) for i in 1:epochs))
trnmini = ctrn[1:20]
dev = collect(ddev)
println("epochs: ", epochs)

@info "Starting Training"
model = train!(model, trnx10, dev, trnmini)

@info "Starting Evaluation"
results = []
real = []
for (x, y) in dev
    push!(results, map( x-> x[1], argmax(model(x); dims=1))...)
    push!(real, y...)
end

Acc = sum(map( x -> x[1] == x[2], zip(real, results))) / length(real)
TN, TP, FP = 0, 0, 0
for (r, p) in zip(real, results)
    TN += (!r && !p)
    TP += (r && p)
    FP += (!r && p)
end
P = TP / (TP + FP)
R = TP / (TP + TN)
F1 = 2 * P * R / ( P + R )

@info "CNN Discriminator model performance report"
@show @sprintf("Accuracy %.2f", Acc)
@show @sprintf("Recall %.2f", R)
@show @sprintf("Precision %.2f", P)
@show @sprintf("F1-Score %.2f", F1)
