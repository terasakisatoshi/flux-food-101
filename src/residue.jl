#=
$ mkdir ~/tmp
$ cd ~/tmp
$ git clone https://github.com/avik-pal/Flux.jl.git
$ cd Flux.jl
$ git checkout -b depthwiseconv origin depthwiseconv
replace ~/.julia/packages/Flux/UHjNa with all of them under ~/tmp/Flux.jl
=#

using Flux
using Flux:Chain
using Flux:@treelike

struct NaiveBottleNeck
    layers::Chain
    stride::Int
end

function NaiveBottleNeck(in_ch::Int, middle_ch::Int, out_ch::Int; stride=1)
    chain = Chain(Conv((1,1), in_ch=>middle_ch, relu),
                  Conv((3,3), middle_ch=>middle_ch, relu, stride=stride, pad=1),
                  Conv((1,1), middle_ch=>out_ch),
                  )
    NaiveBottleNeck(chain, stride)
end

@treelike NaiveBottleNeck

function (m::NaiveBottleNeck)(x)
    h = m.layers(x)
    if size(h)==size(x)
        relu.(h + x)
    else
        relu.(h)
    end
end

#= test
out=NaiveRes(5=>5)(rand(23,30,5,2))
println(size(out))
=#

struct NaiveResNet
    layers::Chain
end

forward() = Chain(Conv((7,7), 3=>32, relu, stride=2, pad=3),
                    NaiveBottleNeck(32, 16, 64),
                    NaiveBottleNeck(64,32,64,stride=2),
                    NaiveBottleNeck(64,32,128),
                    NaiveBottleNeck(128,64,128,stride=2),
                    NaiveBottleNeck(128,64,256),
                    NaiveBottleNeck(256,128,256,stride=2),
                    NaiveBottleNeck(256,128,256),
                    MaxPool((2,2)),
                    x -> reshape(x, :, size(x, 4)),
                    Dense(12544, 2048),
                    Dense(2048, 1024),
                    Dense(1024, 101),
                    softmax
                    )


NaiveResNet() = NaiveResNet(forward())

@treelike NaiveResNet

(model::NaiveResNet)(x) = model.layers(x)


using Flux
using CuArrays

function test_func()
    model = NaiveResNet()
    @show size(model(rand(224, 224, 3, 2)))
    model = model |> gpu
    @show size(model(rand(224, 224, 3, 32)|> gpu))
    batchsize=64
    loss(x,y)= Flux.crossentropy(model(x), y)
    total_loss=0
    for i in 1:500
        X = rand(224,224,3,batchsize) |> gpu
        Y = Flux.onehotbatch([rand(1:101) for i in 1:batchsize], 1:101) |> gpu
        Flux.testmode!(model)
        total_loss += loss(X, Y).data
    end
end

#test_func()