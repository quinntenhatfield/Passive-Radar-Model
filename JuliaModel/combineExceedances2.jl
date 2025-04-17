function combineExceedances2(raw, rThresh, dThresh)
    # If raw is empty, return an empty array
    if isempty(raw)
        return []
    end

    # Extract unique directions (second element of each tuple)
    dops = unique([r[2] for r in raw])
    
    # Initialize an empty list to store the "strings" of exceedances
    strings = []

    # Loop over each unique direction
    for ii in 1:length(dops)
        # Filter raw data for the current direction
        rngs = sort([r for r in raw if r[2] == dops[ii]], by = x -> x[1])
        
        # Calculate the range differences
        rDif = diff([r[1] for r in rngs])
        
        # Find the indices where the range difference exceeds the threshold
        idx = findall(x -> x > rThresh, rDif)
        
        # If no exceedance is found, use the entire sequence
        if isempty(idx)
            idx = [0, length(rngs)]
        else
            idx = [0; idx; length(rngs)]
        end
        
        # Create strings of exceedances
        for jj in 1:length(idx) - 1
            # Create a new string from the range between idx(jj) and idx(jj+1)
            str = rngs[idx[jj] + 1:idx[jj + 1]]
            
            # Set the second column (direction) to the current direction
            for s in str
                s = (s[1], dops[ii])  # Modify tuple
            end
            
            # Append to the strings list
            push!(strings, str)
        end
    end

    # Now, group strings into clusters
    clusters = [strings[1]]  # Initialize with the first string
    
    # Loop through each string
    for ii in 2:length(strings)
        matchFound = false
        
        # Loop through each cluster
        for jj in 1:length(clusters)
            # Initialize a flag to check for overlap
            overlap = false
            
            # Compare each range in the current cluster with each range in the current string
            for r1 in [r[1] for r in clusters[jj]]
                for r2 in [r[1] for r in strings[ii]]
                    # Calculate range difference and check if it is within the threshold
                    if abs(r1 - r2) < rThresh && abs(clusters[jj][1][2] - strings[ii][1][2]) < dThresh
                        overlap = true
                        break
                    end
                end
                if overlap
                    break
                end
            end
            
            # If there's an overlap, add the string to the cluster
            if overlap
                append!(clusters[jj], strings[ii])
                matchFound = true
                break
            end
        end
        
        # If no match is found, create a new cluster
        if !matchFound
            push!(clusters, strings[ii])
        end
    end
    
    # Return the clusters
    return clusters
end
