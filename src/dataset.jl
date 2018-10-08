using Random
using Images, CoordinateTransformations

#taken from
#https://github.com/FluxML/Metalhead.jl/blob/b3c09e58ffa907ab1ef8946a6f9e45858bea0a93/src/utils.jl#L28
function preprocess(img)
    img = channelview(imresize(img, (224, 224))) .* 255
    img .-= [123.68, 116.779, 103.939]
    img = permutedims(img, (3,2,1))
end

function copyimg(img)
    # to reset index of imageview
    imresize(img,size(img))
end

struct Dataset
    len::Int
    data::Array{Tuple{String,Int64},1}
    augment::Bool
    image_cache::Dict{Int,Array{RGB{Normed{UInt8,8}},2}}
    use_cache::Bool
    function Dataset(data; train=true)
        augment=train
        use_cache=train
        image_cache = Dict{Int,Array{RGB{Normed{UInt8,8}},2}}()
        new(length(data), data, augment, image_cache, use_cache)
    end
end

function get_example(dataset::Dataset, i::Int)
    path, label = dataset.data[i]
    if dataset.use_cache && haskey(dataset.image_cache, i)
        img = dataset.image_cache[i]
    else
        img = load(path)
        dataset.image_cache[i] = img
        #dataset.image_cache[i] = imresize(img, (224, 224))
    end
    img = copyimg(img)
    if dataset.augment
        #scale transform
        img = imresize(img, (rand(368:512), rand(368:512)))
        #crop
        h, w = size(img)
        b_h = rand(1:h - 224)
        b_w = rand(1:w - 224)
        img = img[b_h:b_h+224, b_w:b_w+224]
        tfm = LinearMap(RotMatrix(-pi/2 * rand([0,1,2,3])))
        img = warp(img, tfm)
        img = imresize(img,(224, 224))

        domirror = rand([true, false])
        if domirror
            img = channelview(img)
            newimg = zeros(eltype(img), size(img))
            x_range = size(img)[3]
            for i in 1:x_range
                newimg[:,:,x_range - i + 1] = img[:, :, i]
            end
            img = colorview(RGB, newimg)
        end
    end
    img = preprocess(img)
    return img, label
end

Base.length(dataset::Dataset) = dataset.len
