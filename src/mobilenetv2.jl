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

struct ExpandedConv
    layers::Chain
    stride::Int
end

function ExpandedConv(expand::Int, ch::Pair{<:Integer,<:Integer}; stride=1)
    inch=ch[1]
    outch=ch[2]
    expandedch = inch * expand
    if expand != 1
        chain = Chain(Conv((1,1), inch=>expandedch),
                      BatchNorm(expandedch, relu6),
                      DepthwiseConv((3,3),expandedch,relu6,stride=stride,pad=1),
                      BatchNorm(expandedch, relu6),
                      Conv((1,1),expandedch=>outch),
                      BatchNorm(outch))
    else
        chain = Chain(DepthwiseConv((3,3),expandedch,relu6,stride=stride,pad=1),
                      BatchNorm(expandedch, relu6),
                      Conv((1,1),expandedch=>outch),
                      BatchNorm(outch))
    end
    ExpandedConv(chain, stride)
end

@treelike ExpandedConv

function (ec::ExpandedConv)(x)
    h=ec.layers(x)
    if size(h)==size(x)
        h + x
    else
        h
    end
end

#= test
out=ExpandedConv(3,5=>5)(rand(23,30,5,2))
println(size(out))
=#

struct MobileNetv2
    layers::Chain
end

mv2() = Chain(Conv((3,3),3=>32,stride=2,pad=1),
                    BatchNorm(32,relu6),
                    ExpandedConv(1,32=>16),
                    ExpandedConv(6,16=>24,stride=2),
                    ExpandedConv(6,24=>24),
                    ExpandedConv(6,24=>32,stride=2),
                    ExpandedConv(6,32=>32),
                    ExpandedConv(6,32=>32),
                    ExpandedConv(6,32=>64,stride=2),
                    ExpandedConv(6,64=>64),
                    ExpandedConv(6,64=>64),
                    ExpandedConv(6,64=>64),
                    ExpandedConv(6,64=>96),
                    ExpandedConv(6,96=>96),
                    ExpandedConv(6,96=>96),
                    ExpandedConv(6,96=>160,stride=2),
                    ExpandedConv(6,160=>160),
                    ExpandedConv(6,160=>160),
                    ExpandedConv(6,160=>320),
                    Conv((1,1),320=>120),
                    BatchNorm(120,relu6),
                    MeanPool((7,7)),
                    x -> reshape(x, :, size(x, 4)),
                    Dense(120,101),
                    softmax
                    )


MobileNetv2() = MobileNetv2(mv2())

@treelike MobileNetv2

(mv2::MobileNetv2)(x) = mv2.layers(x)

#=
using Flux
using CuArrays
model = MobileNetv2()
@show size(model(rand(224,224,3,2)))
model = model |> gpu
@show size(model(rand(224,224,3,2)|> gpu))
=#
