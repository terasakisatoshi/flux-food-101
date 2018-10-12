#=
$ mkdir ~/tmp
$ cd ~/tmp
$ git clone https://github.com/avik-pal/Flux.jl.git
$ cd Flux.jl
$ git checkout -b depthwiseconv origin depthwiseconv
replace ~/.julia/packages/Flux/UHjNa with all of them under ~/tmp/Flux.jl
=#

using Flux
using Flux:Chain, DepthwiseConv
using Flux:@treelike

relu6(x) = min(max(zero(x),x), eltype(x)(6))

struct NaiveRes
    layers::Chain
    stride::Int
end

function NaiveRes(ch::Pair{<:Integer,<:Integer}; stride=1)
    inch = ch[1]
    hidden = ch[1]
    outch = ch[2]
    chain = Chain(Conv((3,3), inch=>hidden, relu6, stride=stride, pad=1),
                  Conv((3,3), hidden=>hidden, relu6, stride=1, pad=1),
                  #Conv((3,3), hidden=>hidden, relu6, stride=1, pad=1),
                  Conv((3,3), inch=>outch, stride=1, pad=1),
                  )
    NaiveRes(chain, stride)
end

@treelike NaiveRes

function (ec::NaiveRes)(x)
    h=ec.layers(x)
    if size(h)==size(x)
        h + x
    else
        h
    end
end

#= test
out=NaiveRes(5=>5)(rand(23,30,5,2))
println(size(out))
=#

struct NaiveResNet
    layers::Chain
end

forward() = Chain(Conv((3,3),3=>32, relu6, stride=2,pad=1,),
                    NaiveRes(32=>64),
                    NaiveRes(64=>64,stride=2),
                    NaiveRes(64=>64),
                    NaiveRes(64=>64,stride=2),
                    NaiveRes(64=>64),
                    NaiveRes(64=>64,stride=2),
                    x -> reshape(x, :, size(x, 4)),
                    Dense(12544,2048),
                    Dense(2048,101),
                    softmax
                    )


NaiveResNet() = NaiveResNet(forward())

@treelike NaiveResNet

(model::NaiveResNet)(x) = model.layers(x)


using Flux
using CuArrays
model = NaiveResNet()
@show size(model(rand(224, 224, 3, 2)))
model = model |> gpu
@show size(model(rand(224, 224, 3, 32)|> gpu))
