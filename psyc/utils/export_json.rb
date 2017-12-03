#!/usr/bin/env ruby

require 'rubygems'
require 'json'

DOUBLE_FORMAT = '%.15e'
LAYER_TYPES = %w(layer convolutional pooling recurrent lstm softmax)
CONVOLUTIONAL_PARAMS = [:features_count, :region_size, :stride, :input_width, 
                        :input_height, :output_width, :output_height, :padding]
GENERIC_PARAMS = [:vector_size]
HEADER_INFO = [:flags, :loss_function, :epochs, :batch_count]
LOSS_FUNCTIONS = [:quadratic, :cross_entropy]

data_file = ARGV[0]
if !data_file
    puts "Usage: #{$0} SAVED_NET_FILE [OUTPUT_data_file]"
    exit 1
end
data_file = File.expand_path data_file
data = File.read data_file
lines = data.split /\n+/
header = lines.shift
if (m = header.match(/\-\-v(\d+\.\d+\.\d+),([\d\.,]+)/))
    version, _info = m[1], m[2]
    _info = _info.split(',')
    info = {}
    _info.each_with_index{|n, i|
        key = HEADER_INFO[i] 
        if key
            n = n.strip.to_i
            if key == :loss_function
                func = LOSS_FUNCTIONS[n]
                info[key] = func if func
            else
                info[key] = n
            end
        end
    }
    header = lines.shift
end
netsize, layout_def = header.split ':', 2
if !netsize || !layout_def
    puts "Invalid header"
    puts header
    exit 1
end
netsize = netsize.to_i
if netsize.zero?
    puts "Network size must be > 0"
    exit 1
end
layer_def = '(\d+|\[[\d,]+\])'
layers_rexp = /#{([layer_def] * netsize).join(',')}/
layer_defs = layout_def.match layers_rexp
if !layer_defs
    puts "Invalid layout"
    puts layer_def
    exit 1
end
layout = []
layers = []
netsize.times.each{|i|
    ldef = layer_defs[i + 1] 
    if ldef[/^\[/]
        ldef = ldef.match /\[([\d,]+)\]/ 
        if !ldef
            puts "Invalid layer definition"
            puts ldef
            exit 1
        end
        ldef = ldef[1]
        lheader = []
        lparams = ldef.split(',').map{|par| par.to_i}
        ltype = lparams.shift
        typename = LAYER_TYPES[ltype]
        lheader << typename
        layer = {type: typename}
        if !typename
            puts "Layer[#{i}]: Invalid layer type: #{ltype}"
            exit 1
        end
        argc = lparams.shift
        if !argc || argc.zero?
            puts "Layer[#{i}]: Missing arg. count"
            exit 1
        end
        lsize = lparams.shift
        layer[:size] = lsize
        argc -= 1
        if version
            lflags = lparams.shift
            argc -= 1
        end
        if typename == 'convolutional' || typename == 'pooling'
            args = {}
            argc.times.each{|aidx|
                arg = lparams[aidx]
                if !arg
                    puts "Layer[#{i}]: argument #{aidx} not found!"
                    puts ldef
                    exit 1
                end
                argname = CONVOLUTIONAL_PARAMS[aidx]
                args[argname] = arg
                layer[argname] = arg
            }
            lheader << args
        else
            lheader << lsize 
            if version
                args = {}
                args[:flags] = lflags if lflags
                argc.times.each{|aidx|
                    arg = lparams[aidx] 
                    if !arg
                        puts "Layer[#{i}]: argument #{aidx} not found!"
                        puts ldef
                        exit 1
                    end
                    argname = GENERIC_PARAMS[aidx]
                    if argname
                        args[argname] = arg
                        layer[argname] = arg
                    end
                }
                lheader << args
            end
        end
    elsif ldef[/^\d+$/]
        layout << ldef.to_i
        layer = {type: 'layer', size: ldef.to_i}
        layers << layer
        next
    else
        puts "Invalid layer definition"
        puts ldef
        exit 1
    end
    layout << lheader
    layers << layer
}

lineno = 1
layers[1..-1].each_with_index{|l, i|
    ltype = l[:type] || 'layer'
    puts "Importing Layer[#{i + 1}]: '#{ltype}'"
    next if ltype == 'pooling'
    if ltype == 'convolutional'
        features_count = l[:features_count]
        if !features_count
            puts "Layer[#{i + 1}]: missing features_count"
            exit 1
        end
        size = features_count
        key = :features
        l[:neurons] = []
    else
        size = l[:size]
        if !size
            puts "Layer[#{i + 1}]: missing size"
            exit 1
        end
        key = :neurons
    end
    puts "Size: #{size}"
    neurons = []
    l[key] = neurons
    size.times.each{|idx|
        line = lines.shift 
        lineno += 1
        #puts "Reading line #{lineno} (lines left: #{lines.count})"
        if !line
            puts "Invalid file: no bias/weights line"
            puts "Layer[#{i + 1}]: Line: #{lineno} (item: #{idx})"
            exit 1
        end
        bias, weights = line.split('|', 2)
        if !bias || !weights
            puts "Invalid file: invalid bias/weights line"
            puts(line[0,10]+'...')
            exit 1
        end
        bias = bias.to_f
        weights = weights.split(',').map{|w| w.to_f}
        neuron = {bias: bias, weights: weights}
        neurons << neuron
    }
}
out = {layout: layout, layers: layers}
out[:version] = version if version
out[:network] = info if info 
out_file = ARGV[1] || "./network.#{Time.now.to_i}.json"
File.open(out_file, 'w'){|f|
    f.write out.to_json
}
puts "File written to #{out_file}"

