
# CA-CFAR (1D)
#
# Perform one-dimensional cell-averaging constant false alarm rate
# (CA-CFAR 1D) processing on a range-Doppler map to detect targets

# Helper function to compute the scale parameter, alpha
function computeAlphaScale(pfa::Float64, nBin::Int)
    return nBin * (pfa^(-1/nBin) - 1)
end

# Helper function to compute the threshold
function computeThreshold(alphaScale::Float64, bins::Vector{Complex{Float64}})
    nBin = length(bins)
    return (alphaScale / nBin) * sum(bins)
end

function cfar(rdMap::Matrix{Complex{Float64}}, pfa::Float64,
              nLead::Int64, nGuardLead::Int64,
              nLag::Int64, nGuardLag::Int64,
              debug::Bool=false, useMitre::Bool=false)

    nBin = nLead + nLag

    alphaScale = computeAlphaScale(pfa, nBin)

    nDopBin, nRngBin = size(rdMap)
	
	indxFirstFullSupport = nLead + nGuardLead + 1
	indxLastFullSupport = nRngBin - nLag - nGuardLag
	if indxFirstFullSupport >= indxLastFullSupport
		error("Not enough valid bins for CFAR. Adjust nLead, nLag, nGuardLead, or nGuardLag.")
	end

    thresholdMap = fill(Inf, nDopBin, nRngBin)

    if useMitre
        # Presize a map for the threshold

        indxFirstFullSupport = nLead + nGuardLead + 1
        indxLastFullSupport = nRngBin - nLag - nGuardLag

        println("Performing CFAR...")

        # Create the threshold range-Doppler map
        # Loop over doppler bins
        for iDop in 1:nDopBin
            if mod(iDop, 100) == 0
                println("\tDoppler bin #$iDop (of #$nDopBin)")
            end
            # Loop over range bins (to test)
            for iRng in indxFirstFullSupport:indxLastFullSupport
                idxLead = (iRng - nGuardLead - nLead):(iRng - nGuardLead - 1)
                idxLag = (iRng + nGuardLag + 1):(iRng + nGuardLag + nLag)
                bins = abs.(rdMap[iDop, [idxLead; idxLag]])

                thresholdMap[iDop, iRng] = computeThreshold(alphaScale, bins)
            end
        end
    else
#        C = abs.([zeros(nDopBin, nLead+nGuardLead) rdMap zeros(nDopBin, nLag+nGuardLag)])
#        filt = [ones(nLead); zeros(nGuardLead+nGuardLag+1); ones(nLag)]
#        D = mapslices(x -> conv(x, filt)[1:end-length(filt)+1], C, dims=2)
#        thresholdMap = D[:, (nLead+nGuardLead+nLag+nGuardLag+1):end] .* alphaScale
#        thresholdMap[:, [1:(nLead+nGuardLead); (end-nLag-nGuardLag+1):end]] .= Inf

		# Define the padded array C
		C = abs.(hcat(zeros(nDopBin, nLead + nGuardLead), rdMap, zeros(nDopBin, nLag + nGuardLag))) #changed back from hcat

		# Define the filter array
		f = vcat(ones(nLead), zeros(nGuardLead + nGuardLag + 1), ones(nLag)) #changed from vcat, really should be hcat

		# Apply the filter row-wise (along the 2nd dimension)
		D = [filt(f, sum(f), row) for row in eachrow(C)]	

		#convert to matrix again
		D = hcat(D...)
		
		#make sizing correct
		D = transpose(D)

		# Apply the filter along the second dimension (columns) with the filter normalized by sum(filt)
		#D_padded = [conv(vcat(zeros(length(filt) ÷ 2), C[i, :], zeros(length(filt) ÷ 2)), filt ./ sum(filt)) for i in 1:size(C, 1)]
		#D = hcat(D_padded...)'  # Reassemble into a matrix and transpose back
		# Define thresholdMap with appropriate slicing and scaling
		# Slice `D` to ensure it aligns with `rdMap` dimensions
		#D = D[:, (nLead + nGuardLead + nLag + nGuardLag + 1):(size(rdMap, 2) + nLead + nGuardLead + nLag + nGuardLag)]
		
		# Scale to create the thresholdMap
		#thresholdMap = D .* alphaScale	
		thresholdMap = D[:, (nLead+nGuardLead+nLag+nGuardLag+1):end] .* alphaScale
		thresholdMap[:, [1:(nLead+nGuardLead); (end-nLag-nGuardLag+1):end]] .= Inf;
		
		
		##next try with filt()
		#filter = [ones(1, nLead) zeros(1, nGuardLead + nGuardLag + 1) ones(1, nLag)] #changed from vcat, really should be hcat
		#filter = vec(filter)
		#a = Int(sum(filter))
		#D = [filt(filter, a, row) for row in eachrow(C)]
		#D = hcat(D...)
		#D = permutedims(D)
		#thresholdMap = D[:, (nLead + nGuardLead + nLag + nGuardLag + 1) : end] .* alphaScale


		
		# Set the first and last columns to `Inf` based on the lead and lag values
		#thresholdMap[:, 1:(nLead + nGuardLead)] .= Inf
		#thresholdMap[:, (end - nLag - nGuardLag + 1):end] .= Inf
    end

    # Find threshold crossings
    detections = findall(abs.(rdMap) .> abs.(thresholdMap))
    detections = [(d[2], d[1]) for d in detections]  # Swap columns to match MATLAB output
	
#	indices = findall(abs.(rdMap) .> abs.(thresholdMap))
#	dop, rng = map(x -> x[1], indices), map(x -> x[2], indices)

	# If rdMap has only one row, transpose `dop` and `rng`
#	if size(rdMap, 1) == 1
#		dop = collect(dop)'
#		rng = collect(rng)'
#	end

#	# Combine rng and dop into a matrix of detections
#	detections = hcat(rng, dop)
	

    # Visualize results
    if debug > 0
        p1 = heatmap(20 .* log10.(abs.(thresholdMap)), xlabel="Range Bin", ylabel="Doppler Bin", title="Threshold Map")
        p2 = heatmap(20 .* log10.(abs.(rdMap)), xlabel="Range Bin", ylabel="Doppler Bin", title="Range-Doppler Map")
        scatter!(p2, [d[1] for d in detections], [d[2] for d in detections], color=:red, markersize=8)
        plot(p1, p2, layout=(2,1))
    end

    return detections, thresholdMap
end


