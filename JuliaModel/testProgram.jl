
# Constants of light and noise
boltz = 1.380649e-23
c = 2.998e8  # speed of light
totalTime = 0
temp = 2.9e2

# EnodeBOutput parameters
RefChannel = 2  # reference channels based off of LTE Toolbox
DuplexMode = "FDD"  # either FDD or TDD
NCellID = 21  # any valid NCID
TotSubframes = 200

# Processing Options
pulseLength = "slot"  # subframe, halfframe, frame, slot(half of a subframe)
downsamp = false  # should we downsample?
PSScor = false  # if you want to correlate w the PSS signals (must also use halfframe pulse length
antennaType = "dir"  # either 'omni' or 'direct' for reflection antenna
randomTargets = false  # T/F whether you want randomly generated targets or manual
numTar = 4  # number of targets if 'randomTargets' is true

# Simulation Options
simTime = 0
writeVid = false
loadFromFile = 1

if loadFromFile == 1
    # load eNodeBOutput.mat
	using MAT
	matfile = MAT.matopen("Ch4Subf200Bw1o25.mat")
	eNodeBOutput = read(matfile, "eNodeBOutput") #input into eNodeBOutput
	close(matfile) #remember to close file
	eNodeBLength = length(eNodeBOutput)
    sr = 3840000  # depending on what the file was sampled at
	eNodeTime = 1:1:(sr * 0.1)
    noiseCoef = 3
    TotSubframes = TotSubframes
end

# assign the signal to 'eNodeBOutput'
if loadFromFile == 0
    # input bit source:
    in = [1, 0, 0, 1]
#    eNodeBOutput, grid, cfg = lteRMCDLTool(cfg, in)
#    sr = cfg["SamplingRate"]
end

# add noise to the signal (as would normally happen), but keep clean
# signal for transmission purposes. More just to see ZC correlation with noise
noiseStream = (randn(size(eNodeBOutput)) + 1im * randn(size(eNodeBOutput))) * sqrt(boltz * temp * sr / 2) * noiseCoef
eNodeBOutputNoisy = eNodeBOutput + noiseStream


# Noise Control
# if not selected for one of the presets
# noiseCoef = 4

#frameMap = Array{Complex{Float64}}(undef, nPulses, curframeSamp)

# Direct Path Receiver Control
dir = Dict(
    "freq" => 700e6,  # antenna frequency (Hz)
    "antGain" => 5,  # antenna dBi (assumes same tx/rx)
    "power" => 40e3,  # Tx Power (W)
    "temp" => 2.9e2,  # Noise Power controlled by temp (K) (also controlled by sr)
    "noiseCells" => 50, # For noise estimation
	"IQStream" => Array{Complex{Float64}}(undef, length(eNodeBOutput), 1),
	"IQDataMap" => Array{Complex{Float64}}(undef, 5, 4),
	"MatchedMap" => Array{Complex{Float64}}(undef, 5, 4), 
	"InterpMap" => Array{Complex{Float64}}(undef, 5, 4)
)

# Reflected Path Receiver Control
refl = Dict(
    "freq" => 700e6,  # antenna frequency (Hz)
    "antGain" => 5,  # antenna dBi (assumes same tx/rx)
    "power" => 40e3,  # Tx Power (W)
    "temp" => 2.9e2,  # Noise Power controlled by temp (K) (also controlled by sr)
    "noiseCells" => 50,  # For noise estimation
    "antType" => antennaType,
	"IQStream" => Array{Complex{Float64}}(undef, length(eNodeBOutput), 1),
	"IQDataMap" => Array{Complex{Float64}}(undef, 5, 4),
	"MatchedMap" => Array{Complex{Float64}}(undef, 5, 4), 
	"InterpMap" => Array{Complex{Float64}}(undef, 5, 4)
)

# Create Transmitter and Receiver (Static)
# Transmitter
txPos = [-1000, -1000, 100]
txVel = [0, 0, 0]
# Receiver
rxPos = [0, 0, 0]
rxVel = [0, 0, 0]

dirPath = Dict(
    "range" => norm(txPos - rxPos),
    "velocity" => norm(txVel - rxVel)
)

# Target Creation
targets = []  # Target Definition

if randomTargets  # target creation for loop for random targets
    for tar in 1:numTar
        posRange = 5000
        velRange = 25
        accRange = rand()
        push!(targets, Dict(
            "position" => [rand() * posRange, rand() * posRange, rand() * posRange],
            "RCS" => 20,
            "velocity" => [randn() * velRange, randn() * velRange, randn() * velRange],
            "acceleration" => [randn() * accRange, randn() * accRange, randn() * accRange]
        ))
    end
else  # 4 targets written in manually
    push!(targets, Dict(
        "position" => [0, 5000, 5000],
        "RCS" => 20,
        "velocity" => [0, 0, 0],
        "acceleration" => [0, 0, 0]
    ))

    push!(targets, Dict(
        "position" => [5000, 0, 4000],
        "RCS" => 20,
        "velocity" => [4, 50, 50],
        "acceleration" => [0, 0, 0]
    ))

    push!(targets, Dict(
        "position" => [1000, 1000, 5000],
        "RCS" => 20,
        "velocity" => [4, 25, 25],
        "acceleration" => [0, 0, 0]
    ))

    push!(targets, Dict(
        "position" => [2000, 3000, 4000],
        "RCS" => 20,
        "velocity" => [13, 0, 12],
        "acceleration" => [0, 0, 0]
    ))
end

# Find Relative Positioning and Velocity of Targets
for tar in 1:length(targets)
    targets[tar]["rangeTx"] = norm(targets[tar]["position"] - txPos)
    targets[tar]["rangeRx"] = norm(targets[tar]["position"] - rxPos)
    targets[tar]["rangeTotal"] = targets[tar]["rangeTx"] + targets[tar]["rangeRx"]  # range from Tx->Tar->Rx
    dt = targets[tar]["position"] - txPos  # distance in vector form
    dr = targets[tar]["position"] - rxPos
    vt = targets[tar]["velocity"] - txVel  # velocity in vector form
    vr = targets[tar]["velocity"] - rxVel
    targets[tar]["velocityTx"] = dot(dt, vt) / norm(dt)  # velocity with respect to Tx/Rx
    targets[tar]["velocityRx"] = dot(dr, vr) / norm(dr)
end


# Creating PRF info
# each length includes pulsewidht, number of pulses in the amount of
# subframes we sampled, the time of that type of that pulse (pw which is
# redundant), amount of samples for curr sr, and the first section in the signal
if pulseLength == "subframe"
    pw = 1e-3
    nPulses = TotSubframes
    curframeTime = 0.001
    curframeSamp = Int(curframeTime * sr)
    curframe0 = eNodeBOutput[1:curframeSamp, 1]  # clean version of the 1st subframe
    curframe0Noisy = eNodeBOutputNoisy[1:curframeSamp, 1]  # noise

elseif pulseLength == "halfframe"
    pw = 5e-3
    nPulses = TotSubframes / 5
    curframeTime = 0.005
    curframeSamp = Int(curframeTime * sr)
    curframe0 = eNodeBOutput[1:curframeSamp, 1]  # clean
    curframe0Noisy = eNodeBOutputNoisy[1:curframeSamp, 1]  # noise

elseif pulseLength == "frame"
    pw = 10e-3
    nPulses = TotSubframes / 10
    curframeTime = 0.01
    curframeSamp = Int(curframeTime * sr)
    curframe0 = eNodeBOutput[1:curframeSamp, 1]  # clean
    curframe0Noisy = eNodeBOutputNoisy[1:curframeSamp, 1]  # noise

elseif pulseLength == "slot"
    pw = 0.5e-3
    nPulses = TotSubframes
    curframeTime = 0.0005
    curframeSamp = Int(curframeTime * sr)
    curframe0 = eNodeBOutput[1:curframeSamp, 1]
    curframe0Noisy = eNodeBOutputNoisy[1:curframeSamp, 1]  # noise

else
    println("Set Desired Pulselength")
end

# update PRF stuff
prfSpacing = 0
prfStart = 1 / pw
nPulses = Int(nPulses)
prf = prfStart * ones(nPulses)
#prf = prf[randperm(length(prf))]
# not dealing with fractional samples, so we adjust the PRI to the nearest samp
for count in 1:nPulses
    global prf = 1 ./ (round.(sr ./ prf) / sr)
    @assert rem(sr, prf[count]) == 0
end

# Reading LTE For PSS
# all basic timing for 20MHz at sampling rate sr_basic or 30.72 Msps
sr_basic = 30720000
symbTime_basic = 2048  # time of a symbol (samples)
cp_0_basic = 160  # time of cyclic prefix of the 0th symbol
cp_1_basic = 144  # time of cyclic prefix of symbols 1-6
# conversion of symbol timing in sampled
symbTime = Int(symbTime_basic / sr_basic * sr)
#symbTime = Int(round(symbTime))
cp_0 = cp_0_basic / sr_basic * sr
cp_1 = cp_1_basic / sr_basic * sr

# delta = 0  # phase rotation TO BE MEASURED LATER?

# time of clean signal
time = length(eNodeBOutput) / sr  # seconds

subcarriers_ss = floor(symbTime / 2) - 3 * 12 + 5 : floor(symbTime / 2) + 3 * 12 - 5  # gives us the 62 subcarriers of both SS, and the DC in the middle
# above could be written - 31 and + 31



# clean pulse map to be used for transmission
pulseIdx = 0
#frameMap = zeros(nPulses, curframeSamp)
frameMap = Array{Complex{Float64}}(undef, nPulses, curframeSamp)
for pulse in 1:nPulses
    global frameMap[pulse, :] = eNodeBOutput[pulseIdx.+(1:size(frameMap, 2)), 1]
    global pulseIdx = pulseIdx + round(curframeSamp)
end


#initialize IQStream imaginary matrices
dir["IQStream"] = Array{Complex{Float64}}(undef, length(eNodeBOutput), 1)
refl["IQStream"] = Array{Complex{Float64}}(undef, length(eNodeBOutput), 1)
dir["IQDataMap"] = Array{Complex{Float64}}(undef, nPulses, round(curframeSamp))
refl["IQDataMap"] = Array{Complex{Float64}}(undef, nPulses, round(curframeSamp))

# video frame counter
m = 1
# clock to run live sim
#tic()
while totalTime <= simTime

    # Send LTE and recieve direct path with stationary Tx and Rx
	# make variables global
	global eNodeBOutput
	
    # Create the signal as an IQ Stream to match how each receiver may receive it
    global dir["IQStream"] = (randn(length(eNodeBOutput), 1) + 1im * randn(length(eNodeBOutput), 1)) * sqrt(boltz * dir["temp"] * sr / 2) * noiseCoef
    global refl["IQStream"] = (randn(length(eNodeBOutput), 1) + 1im * randn(length(eNodeBOutput), 1)) * sqrt(boltz * dir["temp"] * sr / 2) * noiseCoef

    # Generate the sizes of the IQ Data Maps
    global dir["IQDataMap"] = (zeros(nPulses, round(curframeSamp)) + 1im * zeros(nPulses, round(curframeSamp)))
    global refl["IQDataMap"] = (zeros(nPulses, round(curframeSamp)) + 1im * zeros(nPulses, round(curframeSamp)))

    # direct path
    global eNodeBOutput = transpose(eNodeBOutput)
    local pw = curframeTime
    local pulseIdx = Int(0)
    for pulse in 1:nPulses
        dirPath["pScale"] = 10^((log10(dir["power"]) + 2 * dir["antGain"] + 20 * log10(c / dir["freq"]) - 30 * log10(4 * pi) - 20 * log10(dirPath["range"])) / 20)
        dirPath["doppler"] = -dirPath["velocity"] * dir["freq"] / c
        pStart = Int(round(floor(dirPath["range"] / c * sr)))
        pEnd = Int(round(min(floor(((dirPath["range"] / c) + pw) * sr), length(dir["IQStream"]))))
        if pEnd + pulseIdx > length(dir["IQStream"])  # helps account for overflow of the last LTE pulse
            pEnd = length(dir["IQStream"]) - pulseIdx
        end
        # changed pEnd to go to the whole signal
        if pStart < 1
            continue
        end
        local pFrac = rem(dirPath["range"] / c * sr, 1)
        tResponse = dopShift(frameMap[pulse, :], dirPath["doppler"], sr, sum(1 ./ prf[1:pulse-1])) * dirPath["pScale"] * exp(-1im * 2 * pi * pFrac)
        tResponse = timeShift(tResponse, rem(dirPath["range"] * sr / c, 1))
        dir["IQStream"][pulseIdx .+ (pStart:pEnd)] = dir["IQStream"][pulseIdx .+ (pStart:pEnd)] + tResponse[1:pEnd .- pStart + 1]
        if refl["antType"] == "omni"
            refl["IQStream"][pulseIdx + (pStart:pEnd)] = refl["IQStream"][pulseIdx .+ (pStart:pEnd)] .+ tResponse[1:pEnd - pStart + 1]
        end
        pulseIdx = pulseIdx + Int(round(sr / prf[pulse]))
        global txPos = txPos + txVel / prf[pulse]
        global rxPos = rxPos + rxVel / prf[pulse]
        dirPath["range"] = norm(txPos - rxPos)
    end

    for tar in 1:length(targets)

        pulseIdx = Int(0)
        for pulse in 1:nPulses
            targets[tar]["pScale"] = 10^((10 * log10(refl["power"]) + 2 * refl["antGain"] + targets[tar]["RCS"]
                + 20 * log10(2.998e8 / refl["freq"]) - 30 * log10(4 * pi) - 20 * log10(targets[tar]["rangeTx"])
                - 20 * log10(targets[tar]["rangeRx"])) / 20)
            # Change doppler to frquency shift
            targets[tar]["doppler"] = -(targets[tar]["velocityTx"] + targets[tar]["velocityRx"]) * refl["freq"] / 2.998e8
            # Find the start and stop index for where to insert the pulse
            pStart = Int(round(floor((targets[tar]["rangeTx"] + targets[tar]["rangeRx"]) / 2.998e8 * sr)))
            pEnd = Int(round(min(floor((((targets[tar]["rangeTx"] + targets[tar]["rangeRx"]) / 2.998e8) + pw) * sr), length(refl["IQStream"]))))
            if pEnd + pulseIdx > length(refl["IQStream"])  # helps account for overflow of the last LTE pulse
                pEnd = length(refl["IQStream"]) - pulseIdx
            end
            # pStart should never be less than zero/one
            if pStart < 1
                continue
            end
            local pFrac = rem((targets[tar]["rangeTx"] + targets[tar]["rangeRx"]) / 2.998e8 * sr, 1)
            # Apply doppler shift and phase shift to clean pulse
            tResponse = dopShift(frameMap[pulse, :], targets[tar]["doppler"], sr, sum(1 ./ prf[1:pulse-1])) * targets[tar]["pScale"] * exp(-1im * 2 * pi * pFrac)
            tResponse = timeShift(tResponse, rem((targets[tar]["rangeTx"] + targets[tar]["rangeRx"]) * sr / 2.998e8, 1))

            # Add the modified pulse to the IQ Data Map
            refl["IQStream"][pulseIdx .+ (pStart:pEnd)] = refl["IQStream"][pulseIdx .+ (pStart:pEnd)] .+ tResponse[1:pEnd .- pStart .+ 1]
            pulseIdx = pulseIdx + Int(round(sr / prf[pulse]))

            # Kinematics updates
            global txPos = txPos + txVel / prf[pulse]
            global rxPos = rxPos + rxVel / prf[pulse]
            targets[tar]["position"] = targets[tar]["position"] + targets[tar]["velocity"] / prf[pulse]
            targets[tar]["velocity"] = targets[tar]["velocity"] + targets[tar]["acceleration"] / prf[pulse]

            # Recalculate range and vel (relative)
            targets[tar]["rangeTx"] = norm(targets[tar]["position"] - txPos)
            targets[tar]["rangeRx"] = norm(targets[tar]["position"] - rxPos)
            targets[tar]["rangeTotal"] = targets[tar]["rangeTx"] + targets[tar]["rangeRx"]
            dt = targets[tar]["position"] - txPos
            dr = targets[tar]["position"] - rxPos
            vt = targets[tar]["velocity"] - txVel
            vr = targets[tar]["velocity"] - rxVel
            targets[tar]["velocityTx"] = dot(dt, vt) / norm(dt)
            targets[tar]["velocityRx"] = dot(dr, vr) / norm(dr)

        end
    end

    # Subtract the direct IQStream from the reflected IQStream if omni.
    # May have to multiply direct IQStream by a coefficient to match the power of
    # the pulse being removed if they are uneven due to antenna gain
    if refl["antType"] == "omni"
        refl["IQStream"] = refl["IQStream"] - dir["IQStream"]
    end

    # Pull the IQStream data and place it appropriately in the IQDataMap
    pulseIdx = Int(0)
    # eNodeBOutput = eNodeBOutput.'
    for pulse in 1:nPulses
        dir["IQDataMap"][pulse, :] = dir["IQStream"][pulseIdx .+ (1:size(dir["IQDataMap"], 2)), 1]
        refl["IQDataMap"][pulse, :] = refl["IQStream"][pulseIdx .+ (1:size(refl["IQDataMap"], 2)), 1]
        pulseIdx = pulseIdx + Int(round(curframeSamp))
    end

	global totalTime = 1;

end

# flip streams to match processing 
dir["IQStream"] = transpose(dir["IQStream"])
refl["IQStream"] = transpose(refl["IQStream"])

# Create a Matched filter of the IQ data map. Use the dir IQDataMap to
# match with the refl IQDataMap
dir["MatchedMap"] = Array{Complex{Float64}}(undef, nPulses, curframeSamp)
refl["MatchedMap"] = Array{Complex{Float64}}(undef, nPulses, curframeSamp)

for pulse in 1:nPulses
    if PSScor # use if we want to correlate simply based off of the PSS with the direct path and the reflected path
        dir["MatchedMap"][pulse, :] = ifft(fft(dir["IQDataMap"][pulse, :], size(dir["IQDataMap"], 2)).*conj(fft(zc_time(NCID2, :), size(dir["IQDataMap"], 2))), size(dir["IQDataMap"], 2))
        refl["MatchedMap"][pulse, :] = ifft(fft(refl["IQDataMap"][pulse, :], size(refl["IQDataMap"], 2)).*conj(fft(zc_time(NCID2, :), size(refl["IQDataMap"], 2))), size(refl["IQDataMap"], 2))
        refl["MatchedMap"][pulse, :] = ifft(fft(refl["MatchedMap"][pulse, :], size(refl["MatchedMap"], 2)).*conj(fft(dir["MatchedMap"][pulse, :], size(refl["IQDataMap"], 2))), size(refl["IQDataMap"], 2))
    else
		refl["MatchedMap"][pulse, :] = ifft(fft(refl["IQDataMap"][pulse, :]) .* conj(fft(dir["IQDataMap"][pulse, :])))
		#refl["MatchedMap"][pulse, :] = ifft(fft(refl["IQDataMap"][pulse, :], size(refl["IQDataMap"], 2)) .* conj(fft(dir["IQDataMap"][pulse, :], size(refl["IQDataMap"], 2))), size(refl["IQDataMap"], 2))
    end
end


xaxis = 1:1:Int(round(length(refl["MatchedMap"][1, :])))
f = plot(
	plot(xaxis, abs.(refl["MatchedMap"][1, :])), 
	plot(xaxis, abs.(refl["MatchedMap"][2, :])), 
	plot(xaxis, abs.(refl["MatchedMap"][3, :])), 
	plot(xaxis, abs.(refl["MatchedMap"][4, :])), 
	layout = (4, 1), 
	size = (800, 600)
)
display(f)

refl["MatchedMap"] = refl["MatchedMap"][:, 2:end] # take off one of the samples at the beginning for accuracy 

# CPI Control
bw = sr / 1.536 # Hz max for any LTE signal at a specific sr

# Target Detection
pulseBlanking = true
coherentProcessing = true
runMTI = false
interpFactor = 1
maxRange = 20000
# maxRange = 2.998e8/max(prf)
cellRes = ceil(Int, sr/bw)

constThresh = -95
drawExceed = false # drawExceed uses the const threshold but everything else is CFAR
pfa = 1e-6 # CFAR Parameters

#nLead = 20
#nGuardLead = 5 #nLead / 2 | cellRes*10
#nLag = 20
#nGuardLag = 5 #nLag / 2 | cellRes*10

nLead = 10
nGuardLead = Int(cellRes*10)
nLag = 10
nGuardLag = Int(cellRes*10)

x = [0; cumsum(1 ./ (prf[1:end-1]))] # #ok<UNRCH>
nPulses *= interpFactor
xq = x #range(0, stop=x[end], length=nPulses)

# Optional Windowing
window = ones(length(xq)) # hamming(length(xq)).'
windowLoss = -20 * log10(mean(window))
newPRF = 1 / xq[2]

dir["InterpMap"] = Array{Complex{Float64}}(undef, size(dir["MatchedMap"]))
refl["InterpMap"] = Array{Complex{Float64}}(undef, size(refl["MatchedMap"]))

for idx in 1:size(refl["MatchedMap"], 2)
    v = refl["MatchedMap"][:, idx]
    # Interpolate to allow for doppler processing with staggered PRIs
	interp_func = interpolate((x,), v, Gridded(Linear()))
    refl["InterpMap"][:, idx] = interp_func.(xq) .* window # nearest

    v = refl["MatchedMap"][:, idx]
	interp_func = interpolate((x,), v, Gridded(Linear()))
    refl["InterpMap"][:, idx] = interp_func.(xq) .* window # nearest
end
#RanDopMap = fftshift(fft(refl["InterpMap"], length(xq), 1), 1)
#RanDopMap = fftshift(fft(refl["InterpMap"], dims=1), dims=1)
#RanDopMap = fftshift(fft(refl["InterpMap"], 2))

RanDopMap = fftshift(fft(refl["InterpMap"], 1), 1)

# CFAR
# Run CFAR to get detections
exceedances, thresholdMap = cfar(RanDopMap, pfa, nLead, nGuardLead, nLag, nGuardLag, true, false)
#exceedances, thresholdMap = cfar2(RanDopMap, pfa, nLead, nGuardLead, nLag, nGuardLag)

#heatmap(thresholdMap, xlabel="Range Bin", ylabel="Doppler Bin", title="Threshold Map Debug")

# Calculate range and doppler vectors for the plots
rngVec = range(0, stop=maxRange, length=size(RanDopMap, 2)) # return rdMap compatible range vector (meters)
dopVec = range(-2.998e8/refl["freq"]*newPRF/nPulses*(-nPulses)/2, stop=-2.998e8/refl["freq"]*newPRF/nPulses*(nPulses-2)/2, length=nPulses) # return rdMap compatible doppler vector (Hz)
coherentProcessing = true

# Create clusters for groups of exceedances
if coherentProcessing
    clusters = combineExceedances2(exceedances, cellRes*2, ceil(nPulses/4))
else
    clusters = combineExceedances2(exceedances, cellRes*2, 0)
end

exceedances = zeros(length(clusters), 2)
for count in 1:length(clusters)
    tempVal = Vector{Complex{Float64}}(undef, length(clusters[count]))
    for counter in 1:length(clusters[count])
        tempVal[counter] = RanDopMap[clusters[count][counter][2], clusters[count][counter][1]]
    end
	local maxIdx
    _, maxIdx = findmax(abs.(tempVal))
    exceedances[count, 1] = clusters[count][maxIdx][1]
    exceedances[count, 2] = clusters[count][maxIdx][2]
end

if !coherentProcessing
    clusters = combineExceedances(exceedances, cellRes*2, nPulses)
    exceedances = zeros(length(clusters), 2)
    newCount = 1
    for count in 1:length(clusters)
        if size(clusters[count], 1) < nPulses/2
            continue
        end
        global tempVal = zeros(size(clusters[count], 1))
        for counter in 1:size(clusters[count], 1)
            tempVal[counter] = RanDopMap[clusters[count][counter][2], clusters[count][counter][1]]
        end
		local maxIdx
        _, maxIdx = findmax(abs.(tempVal))
        exceedances[newCount, 1] = clusters[count][maxId][1]
        exceedances[newCount, 2] = clusters[count][maxIdx][2]
        global newCount += 1
    end
    exceedances = exceedances[1:newCount-1, :]
end
exceedances = Int.(exceedances)

# Get the power of each exceedance
value = zeros(ComplexF64, size(exceedances, 1))
for count in 1:size(exceedances, 1)
    value[count] = RanDopMap[exceedances[count, 2], exceedances[count, 1]]
end

# Get the max power exceedance from each cluster
_, idx = findmax(abs.(value))
maxIdx = exceedances[idx, 1]
maxVal = value[idx]

# Prepare constants to do interpolation between cells
rngCell = rngVec[2] - rngVec[1]
binDiff = abs.(mean(diff(dopVec)))
# Sort exceedances by range
sortExceed = sort(exceedances[:, 1], rev=true)
exceedances = sort!(exceedances, dims = 1, rev=true)

adjustedExceedances = zeros(size(exceedances))

println("Detections")
for count in 1:length(sortExceed)
    # Get total power of relevant doppler cells
    dopInterp = abs.(RanDopMap[exceedances[count, 2], exceedances[count, 1]])
    if exceedances[count, 2] == nPulses || (exceedances[count, 2] > 1 && abs.(RanDopMap[exceedances[count, 2]-1, exceedances[count, 1]]) > abs.(RanDopMap[exceedances[count, 2]+1, exceedances[count, 1]]))
        dopInterp += abs.(RanDopMap[exceedances[count, 2]-1, exceedances[count, 1]])
        idxArr = [exceedances[count, 2]-1, exceedances[count, 2]]
    else
        dopInterp += abs.(RanDopMap[exceedances[count, 2]+1, exceedances[count, 1]])
        idxArr = [exceedances[count, 2], exceedances[count, 2]+1]
    end
    # Interpolate with nearest neighbor
    dopArr = 0
    for idx in idxArr
        dopArr += dopVec[idx] * abs.(RanDopMap[idx, exceedances[count, 1]]) / dopInterp
    end
    # Adjust the velocity (and range?) to match the interp value
    adjustedExceedances[count, 1] = exceedances[count, 1] * 2.998e8 / sr
    adjustedExceedances[count, 2] = dopArr
    # Print the detection data
	if adjustedExceedances[count, 2] < 100 && adjustedExceedances[count, 2] > -100
		println("Range: ", exceedances[count, 1] * 2.998e8 / sr + dirPath["range"], "m\tPower: ", 20 * log10(abs.(value[count])), " dBW\tVelocity: ", dopArr, " m/s")
	end
end

# Sort the truth data by range
#_, sortTar = sort(vcat(targets["rangeTotal"]), rev=true)
sortTar = sort(collect(targets), by=x -> x["rangeTotal"], rev=true)
println("Targets")
for count in 1:length(sortTar)
    # Calculate loss from between doppler cells
    binOff, binIdx = findmin(abs.(dopVec .- (mod((targets[count]["velocityTx"] + targets[count]["velocityRx"]) - dopVec[1], dopVec[1] - dopVec[end] + binDiff) - dopVec[1])))
    dopLoss = 10 * log10(binDiff / (binDiff - binOff))
    # Calculate the expected power return (validates our equation)
    pReturn = 10 * log10(refl["power"]) + 2 * refl["antGain"] + targets[count]["RCS"] + 20 * log10(2.998e8 / refl["freq"]) - 30 * log10(4 * π) - 20 * log10(targets[count]["rangeTx"]) - 20 * log10(targets[count]["rangeRx"]) + 20 * log10(nPulses) + 20 * log10(sr * pw) - dopLoss - windowLoss
    # Display the truth data
    println("Range: ", sortTar[count]["rangeRx"] + sortTar[count]["rangeTx"], "m\tPower: ", pReturn, " dBW\tVelocity: ", sortTar[count]["velocityTx"] + sortTar[count]["velocityRx"], " m/s")
end

println("Direct")
println("Range: ", dirPath["range"], "m") #  'Velocity: ' num2str(dirPath["doppler"]) ' m/s'])
#using PyPlot
#using MeshGrid
#using MeshGrid
# Display the range doppler map


using Plots
using LinearAlgebra

# Display the range Doppler map
fig = 124

# Create X and Y for the mesh grid
X = (1:size(RanDopMap, 2)) .* (2.998e8 / sr)
Y = -2.998e8 / refl["freq"] * newPRF / nPulses .* ((-nPulses / 2):(nPulses - 2) / 2)

# Set color limits
if value === nothing
    nPwr = -200
else
    nPwr = 20 * log10(minimum(abs.(value))) # nPwr goes to power of minimum detection
end

color_limits = (nPwr - 80, nPwr + 80)

# Plot the surface
p = plot(
    X, Y, 20 * log10.(abs.(RanDopMap)),
    st=:surface, c=:viridis,
    xlabel="Range", ylabel="Velocity",
    zlabel="Power (dB)", colorbar=true, zlims=color_limits,
	camera=(60, 20)
)

# Set axes limits
xlims!(0, maxRange)
ylims!(dopVec[end], dopVec[1])
zlims!(color_limits)

# Convert real-world coordinates to indices
range_indices = round.(Int, (adjustedExceedances[:, 2] .- minimum(X)) ./ step(X))
velocity_indices = round.(Int, (adjustedExceedances[:, 1] .- minimum(Y)) ./ step(Y))

# Ensure indices are within bounds
range_indices = clamp.(range_indices, 1, size(RanDopMap, 2))
velocity_indices = clamp.(velocity_indices, 1, size(RanDopMap, 1))

# Extract power at exceedance points
plotPower = [20 * log10(abs(RanDopMap[v, r])) for (v, r) in zip(velocity_indices, range_indices)]

# Highlight exceedances
scatter3d!(
    adjustedExceedances[:, 2], # Range values (original coordinates)
    adjustedExceedances[:, 1], # Velocity values (original coordinates)
    plotPower,
    marker=:circle, color=:red
)
