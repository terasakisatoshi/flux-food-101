include("util.jl")
include("./dataset.jl")
include("./iterator.jl")
include("residue.jl")

using Printf
using Flux
using Statistics
using Flux: onehotbatch, onecold, crossentropy
using Base.Filesystem
using Base.Iterators: partition
using BSON: @load, @save
using CuArrays

function define_model()
    model = NaiveResNet()
    return model
end

function get_dataset(datasetdir)
    train_paris, val_pairs = make_trainval_pairs(datasetdir)
    train_dataset = Dataset(train_paris)
    val_dataset = Dataset(val_pairs, train=false)
    return train_dataset, val_dataset
end

function main()
    datasetdir = expanduser("~/dataSSD120GB/food-101")
    batchsize = 32
    epochs = 100
    cache_multiplier = 1
    train_dataset, val_dataset = get_dataset(datasetdir)

    model = define_model() |> gpu
    if isfile("checkpoint_weights.bson")
        println("loading checkpoin file")
        @load "checkpoint_weights.bson" checkpoint_weights
        Flux.loadparams!(model, checkpoint_weights)
    end
    loss(x,y)= crossentropy(model(x), y)
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
    optimizer = ADAM(params(model))
    for e in 1:epochs
        println("train loop ", e," / ",epochs)
        train_iter = SerialIterator(train_dataset, cache_multiplier * batchsize)
        val_iter = SerialIterator(val_dataset, cache_multiplier * batchsize, shuffle=false)
        total_loss = Float32(0.0)
        total_acc = Float32(0.0)
        cnt = 0
        for (i, batch) in enumerate(train_iter)
            println("progress ", i," / ", floor(Int, length(train_dataset) / batchsize / cache_multiplier))
            X = cat([img for (img, _) in batch]..., dims=4)
            Y = onehotbatch([label for (_, label) in batch], 1:101)
            data = [(X[:,:,:,b] |> gpu, Y[:,b] |> gpu) for b in partition(1: cache_multiplier * batchsize, batchsize)]
            Flux.train!(loss, data, optimizer)
            Flux.testmode!(model)
            for (X,Y) in data
                total_loss += loss(X, Y).data
                total_acc  += accuracy(X, Y)
                cnt += 1
            end
            Flux.testmode!(model, false)
        end
        total_loss /= cnt
        total_acc /= cnt
        @printf("loss = %.3f\n", total_loss)
        @printf("acc = %.3f\n", total_acc)

        println("check accuracy")
        total_loss = Float32(0.0)
        total_acc = Float32(0.0)
        cnt = 0
        Flux.testmode!(model)
        for (i, batch) in enumerate(val_iter)
            println("progress ", i," / ", floor(Int, length(val_dataset) / batchsize / cache_multiplier))
            X = cat([img for (img, _) in batch]..., dims=4)
            Y = onehotbatch([label for (_, label) in batch], 1:101)
            data = [(X[:,:,:,b], Y[:,b]) for b in partition(1: cache_multiplier * batchsize, batchsize)]
            for (X,Y) in data
                total_loss += loss(X, Y).data
                total_acc  += accuracy(X, Y)
                cnt += 1
            end
        end
        Flux.testmode!(model, false)
        total_loss /= cnt
        total_acc /= cnt
        @printf("loss = %.3f\n", total_loss)
        @printf("acc = %.3f\n", total_acc)
        checkpoint = model |> cpu
        checkpoint_weights = Tracker.data.(params(checkpoint))
        @save "checkpoint.bson" checkpoint
        @save "checkpoint_weights.bson" checkpoint_weights

    end
    pretrained = model |> cpu
    weights = Tracker.data.(params(pretrained))
    @save "pretrained.bson" pretrained
    @save "weights.bson" weights
    println("Finished to train")
end

#main()
