include("char_lmodel.jl")

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

@info "Training Language Model"
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