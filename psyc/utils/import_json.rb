#!/usr/bin/env ruby

require 'rubygems'
require 'json'

DOUBLE_FORMAT = '%.15e'
LAYER_TYPES = %w(layer convolutional pooling recurrent lstm softmax)
CONVOLUTIONAL_PARAMS = [:features_count, :region_size, :stride, :input_width, 
                        :input_height, :output_width, :output_height, :padding]

json_file = ARGV[0]
if !json_file
    puts "Usage: #{$0} JSON_FILE [OUTPUT_FILE]"
    exit 1
end
json_file = File.expand_path json_file
net = JSON.parse(File.read(json_file), symbolize_names: true)
if !net.is_a? Hash
    puts "Invalid file!"
    exit 1
end
version = net[:version]
info = net[:network]
layout = net[:layout]
if !layout
    puts "Missing layout!"
    exit 1
end
layers = net[:layers]
if !layers
    puts "Missing :layers!"
    exit 1
end
lidx = -1
input_size = nil
ltypes = []
last_size = nil
layout = layout.map{|l|
    lidx += 1
    if l.is_a? Array
        ltype = l[0]
        if !ltype.is_a? String
            puts "Layer[#{lidx}]: type must be a String, but is #{ltype}"
            exit 1
        end
        typename = ltype.dup
        ltype = LAYER_TYPES.index ltype
        ltypes[lidx] = typename
        if !ltype
            puts "Layer[#{lidx}]: unsupported layer type #{typename}"
            exit 1
        end
        if typename == 'layer'
            lsize, iptsize = l[1, 2]
            input_size ||= iptsize
            last_size = lsize.to_i
            lsize
        elsif %w(convolutional pooling).include?(typename)
            params = l[1]
            if !params
                puts "Layer[#{lidx}]: missing convolutional params"
                exit 1
            end
            argc = CONVOLUTIONAL_PARAMS.length
            args = Array.new argc, 0
            max_arg = 0
            params[:features_count] ||= layers[lidx][:features_count]
            if !params[:features_count]
                puts "Layer[#{lidx}]: missing features count"
                exit 1
            end
            params[:stride] ||= 1
            params[:padding] ||= 0
            if !params[:input_width]
                params[:input_width] = Math.sqrt(last_size)
                params[:input_height] = Math.sqrt(last_size)
            end
            params[:output_width] ||= 0
            params[:output_height] ||= 0
            params.each{|par, val|
                pidx = CONVOLUTIONAL_PARAMS.index par 
                next if !pidx
                max_arg = pidx if pidx > max_arg
                args[pidx] = val.to_i
            }
            args = args[0..max_arg]
            argc = args.length + 1
            lsize = layers[lidx][:size].to_i
            last_size = lsize
            args = [ltype, argc, lsize] + args
            args.inspect
        end
    else
        last_size = l.to_i
        l
    end
}
out = "#{layout.length}:" + layout.join(",") + "\n"
layers[1..-1].each_with_index{|l, i|
    #size = l[:size]
    ltype = ltypes[i + 1]
    next if ltype == 'pooling'
    if ltype == 'convolutional'
        features = l[:features]
        if !features
            puts "Layer[#{i + 1}]: missing features"
            exit 1
        end
        neurons = features
    else
        neurons = l[:neurons]
    end
    puts "Layer[#{i + 1}]: Exporting #{neurons.size} neurons"
    lstr = neurons.map{|n|
        bias = DOUBLE_FORMAT % n[:bias]
        weights = n[:weights].map{|w| DOUBLE_FORMAT % w}
        "#{bias}|#{weights.join(',')}"
    }.join("\n")
    lstr << "\n"
    out << lstr
}
out_file = ARGV[1] || "/tmp/network.#{Time.now.to_i}.data"
File.open(out_file, 'w'){|f|
    f.write out
}
puts "File written to #{out_file}"

