#include("myPackages.jl)
include("myFunctions.jl")
include("read_iq.jl")
include("combineExceedances2.jl")
include("cfar.jl")
using Plots, FFTW, LinearAlgebra, Statistics
using Random, Interpolations, Printf, Base.Threads, DSP


# Constants of light and noise
boltz = 1.380649e-23
c = 2.998e8  # speed of light
totalTime = 0
temp = 2.9e2

# EnodeBOutput parameters
RefChannel = 2  # reference channels based off of LTE Toolbox
DuplexMode = "FDD"  # either FDD or TDD
NCellID = 21  # any valid NCID

# Processing Options
pulseLength = "halfframe"  # subframe, halfframe, frame, slot(half of a subframe)
downsamp = false  # should we downsample?
PSScor = false  # if you want to correlate w the PSS signals (must also use halfframe pulse length
antennaType = "dir"  # either 'omni' or 'direct' for reflection antenna

# Simulation Options
simTime = 0
writeVid = false

# File Inputs
#ref_file = "data/FM_20MPH_97700000Hz_2000000sps_10.0dB_12025_04_01_16-51-55.cfile"
ref_file = "data/LTE_20MPH_1955000000Hz_2000000sps_10.0dB_12025_04_01_17-00-16.cfile"
#sur_file = "data/FM_20MPH_97700000Hz_2000000sps_10.0dB_22025_04_01_16-51-55.cfile"
sur_file = "data/LTE_20MPH_1955000000Hz_2000000sps_10.0dB_22025_04_01_17-00-16.cfile"

sr = 2000000  # depending on what the file was sampled at
c_freq = 97.7e6

# Load in signals
ref_sig = read_iq(ref_file)
sur_sig = read_iq(sur_file)

ref_length = length(ref_sig)
sur_length = length(sur_sig)
ref_total_time = ref_length / sr
sur_total_time = sur_length / sr
ref_subframes = ref_total_time * 10e2
sur_subframes = sur_total_time * 10e2
TotSubframes = Int(floor(min(sur_subframes,ref_subframes)))
shaveTime = Int(TotSubframes / 10e2 * sr)

#shave a partial subframe off end (might help later)
ref_sig = ref_sig[1:shaveTime]
sur_sig = sur_sig[1:shaveTime]



halfSec = 1:1:Int(sr * 0.5)

ref_sig_half_sec = ref_sig[1:Int(sr/2)]
sur_sig_half_sec = ref_sig[1:Int(sr/2)]

# Can convert to a smaller signal here if want to. 

## plot clean and noisy signal for reference
using Plots
sig = plot(
	plot(halfSec, abs.(ref_sig_half_sec), title="Reference Signal"),
	plot(halfSec, abs.(sur_sig_half_sec), title="Surveillance Signal"),
	layout = (2, 1), 
	legend = false
)
display(sig)

# plot the clean and noisy signals in the frequency domain
ref_sig_fft = fftshift(fft(ref_sig_half_sec))
sur_sig_fft = fftshift(fft(sur_sig_half_sec))

sig_fft = plot(
	plot(1:1:Int(length(ref_sig_fft)), abs.(ref_sig_fft), title="Referance Signal"),
	plot(1:1:Int(length(sur_sig_fft)), abs.(sur_sig_fft), title="Surveillance Signal"), 
	layout = (2, 1), 
	legend = false
)
display(sig_fft)



#### Option to downsample eNodeBOutput to just the 6 RB of the PSS/SSS
#if downsamp
#    enb = Dict()
#    enb["NDLRB"] = 6  # number of DL resource blocks
#    ofdmInfo = lteOFDMInfo(setfield(enb, "CyclicPrefix", "Normal"))
#    if sr != ofdmInfo["SamplingRate"]
#        if sr < ofdmInfo["SamplingRate"]
#            @warn("The received signal sampling rate ($(sr/1e6)Ms/s) is lower than the desired sampling rate for cell search / MIB decoding ($(ofdmInfo['SamplingRate']/1e6)Ms/s); cell search / MIB decoding may fail.")
#        end
#        @printf("\nResampling from %0.3fMs/s to %0.3fMs/s for cell search / MIB decoding...\n", sr/1e6, ofdmInfo["SamplingRate"]/1e6)
#    else
#        @printf("\nResampling not required; received signal is at desired sampling rate for cell search / MIB decoding (%0.3fMs/s).\n", sr/1e6)
#    end
#    # Downsample received signal
#    nSamples = ceil(ofdmInfo["SamplingRate"] / round(sr) * size(eNodeBOutput, 1))
#    nRxAnts = size(eNodeBOutput, 2)
#    downsampled = zeros(nSamples, nRxAnts)
#    for i in 1:nRxAnts
 #       downsampled[:, i] = resample(eNodeBOutput[:, i], ofdmInfo["SamplingRate"], round(sr))
#    end
#    noiseCoef = noiseCoef / 2
#    eNodeBOutput = downsampled
#    sr = ofdmInfo["SamplingRate"]
#end

# Creating PRF info
# each length includes pulsewidht, number of pulses in the amount of
# subframes we sampled, the time of that type of that pulse (pw which is
# redundant), amount of samples for curr sr, and the first section in the signal
if pulseLength == "subframe"
    pw = 1e-3
    nPulses = TotSubframes
    curframeTime = 0.001
    curframeSamp = Int(curframeTime * sr)
    curframe0 = ref_sig[1:curframeSamp, 1]  # clean version of the 1st subframe
	curframe1 = sur_sig[1:curframeSamp, 1]  

elseif pulseLength == "halfframe"
    pw = 5e-3
    nPulses = Int(round(TotSubframes / 5))
    curframeTime = 0.005
    curframeSamp = Int(curframeTime * sr)
    curframe0 = ref_sig[1:curframeSamp, 1]  # clean
	curframe1 = sur_sig[1:curframeSamp, 1]

elseif pulseLength == "frame"
    pw = 10e-3
    nPulses = Int(round(TotSubframes / 10))
    curframeTime = 0.01
    curframeSamp = Int(curframeTime * sr)
    curframe0 = ref_sig[1:curframeSamp, 1]  # clean
	curframe1 = sur_sig[1:curframeSamp, 1]

elseif pulseLength == "slot"
    pw = 0.5e-3
    nPulses = Int(TotSubframes * 2)
    curframeTime = 0.0005
    curframeSamp = Int(curframeTime * sr)
    curframe0 = ref_sig[1:curframeSamp, 1]
	curframe1 = sur_sig[1:curframeSamp, 1]

else
    println("Set Desired Pulselength")
end

nPulses = Int(floor(nPulses)) #floor bc don't want too many pulses
# update PRF stuff
prfSpacing = 0
prfStart = 1 / pw
prf = prfStart * ones(Int(nPulses))
#prf = prf[randperm(length(prf))] #this is an error
# not dealing with fractional samples, so we adjust the PRI to the nearest samp
for count in 1:nPulses
    global prf = 1 ./ (round.(sr ./ prf) / sr)
    @assert rem(sr, prf[count]) == 0
end

# Direct Path Receiver Control
dir = Dict(
    "freq" => c_freq,  # antenna frequency (Hz)
    "antGain" => 5,  # antenna dBi (assumes same tx/rx)
    "power" => 40e3,  # Tx Power (W)
    "temp" => 2.9e2,  # Noise Power controlled by temp (K) (also controlled by sr)
    "noiseCells" => 50, # For noise estimation
	"IQStream" => Array{Complex{Float64}}(undef, ref_length, 1),
	"IQDataMap" => Array{Complex{Float64}}(undef, nPulses, round(curframeSamp))
)

# Reflected Path Receiver Control
refl = Dict(
    "freq" => c_freq,  # antenna frequency (Hz)
    "antGain" => 5,  # antenna dBi (assumes same tx/rx)
    "power" => 40e3,  # Tx Power (W)
    "temp" => 2.9e2,  # Noise Power controlled by temp (K) (also controlled by sr)
    "noiseCells" => 50,  # For noise estimation
    "antType" => antennaType,
	"IQStream" => Array{Complex{Float64}}(undef, sur_length, 1),
	"IQDataMap" => Array{Complex{Float64}}(undef, nPulses, round(curframeSamp))

)

# Transmitter Position (estimated)
txPos = [-1000, -1000, 100]
txVel = [0, 0, 0]
# Receiver Position (estimated
rxPos = [0, 0, 0]
rxVel = [0, 0, 0]

dirPath = Dict(
    "range" => norm(txPos .- rxPos), #this is an error of norm()
    "velocity" => norm(txVel .- rxVel)
)



# Reading LTE For PSS
# all basic timing for 20MHz at sampling rate sr_basic or 30.72 Msps
sr_basic = 30720000
symbTime_basic = 2048  # time of a symbol (samples)
cp_0_basic = 160  # time of cyclic prefix of the 0th symbol
cp_1_basic = 144  # time of cyclic prefix of symbols 1-6
# conversion of symbol timing in sampled
symbTime = Int(round(symbTime_basic / sr_basic * sr))
#symbTime = Int(round(symbTime))
cp_0 = cp_0_basic / sr_basic * sr
cp_1 = cp_1_basic / sr_basic * sr

# delta = 0  # phase rotation TO BE MEASURED LATER?

# time of clean signal (not sure if used anywhere close to this)
ref_time = ref_length / sr  # seconds
sur_time = sur_length / sr  # seconds

subcarriers_ss = Int(floor(symbTime / 2) - 3 * 12 + 5) : Int(floor(symbTime / 2) + 3 * 12 - 5)  # gives us the 62 subcarriers of both SS, and the DC in the middle
# above could be written - 31 and + 31

# create the 3 time domain ZC sequences corresponding to 0,1,2 NCID2s
##come back to this later
zc_u = [25, 29, 34]  # u values in the Zadoff-Chu sequence.
zc_time = zeros(ComplexF64, 3, symbTime)
zc_all = zeros(ComplexF64, 3, symbTime)
for u in 1:3
    zc = zadoffChu(zc_u[u])
    local zc_all = zeros(ComplexF64, 3, symbTime)
    zc_all[u, subcarriers_ss] = zc
    zc_time[u, :] = ifft(fftshift(zc_all[u, :]))
end


N = size(curframe0, 1)  # Total length of IQ data
num_rows = size(zc_time, 1)  # Number of rows (should be 3)
num_cols = size(zc_time, 2)  # Number of columns in original ZC sequence

# Zero-pad along columns to match curframe0's width
zc_padded = hcat(zc_time, zeros(num_rows, N - num_cols))

# Compute FFT
curframe0_fft = fft(curframe0)  # FFT along time axis 
curframe1_fft = fft(curframe1)
zc_fft = fft(zc_padded)  # FFT along time axis 

# Perform the matched filter
ref_zc_corr = zeros(ComplexF64, 3, N)
ref_zc_corr[1, :] = ifft(curframe0_fft .* conj.(zc_fft)[1, :])  # IFFT along columns
ref_zc_corr[2, :] = ifft(curframe0_fft .* conj.(zc_fft)[2, :])  # IFFT along columns
ref_zc_corr[3, :] = ifft(curframe0_fft .* conj.(zc_fft)[3, :])  # IFFT along columns

sur_zc_corr = zeros(ComplexF64, 3, N)
sur_zc_corr[1, :] = ifft(curframe1_fft .* conj.(zc_fft)[1, :])  # IFFT along columns
sur_zc_corr[2, :] = ifft(curframe1_fft .* conj.(zc_fft)[2, :])  # IFFT along columns
sur_zc_corr[3, :] = ifft(curframe1_fft .* conj.(zc_fft)[3, :])  # IFFT along columns


max_values0 = [findmax(abs, row) for row in eachrow(ref_zc_corr)]
max_values1 = [findmax(abs, row) for row in eachrow(sur_zc_corr)]


# Extract the values and indices separately
max_magnitudes0 = [val[1] for val in max_values0]  # Maximum magnitudes
max_indices0 = [val[2] for val in max_values0]  
max_mag0 = findmax(max_magnitudes0)
max_index0 = max_indices0[max_mag0[2]]
NCID20 = max_mag0[2]
max_mag0 = max_mag0[1]

max_magnitudes1 = [val[1] for val in max_values1]  # Maximum magnitudes
max_indices1 = [val[2] for val in max_values1]  
max_mag1 = findmax(max_magnitudes1)
max_index1 = max_indices1[max_mag1[2]]
NCID21 = max_mag1[2]
max_mag1 = max_mag1[1]

# Find NCID2 from peak corr and find sample and time offset
#pssStart = corrTimingPeakSamp[NCID2] / sr
#NCID2 = NCID2 - 1  # technically the NCID  values are 0-2 instead of 1-3
# but not useful if using NCID2 later.. just keep it in mind

t_corr = 0:1/sr:curframeTime - 1/sr  # time of correlation for x axis values

# first frame (any length) correlated with the 3 zadoff-chu sequences

ref_ZadoffChuPlot = plot(
	plot(t_corr, abs.(ref_zc_corr[1, :]), ylim = (0, max_mag0), title = "NID = 0"),  
	plot(t_corr, abs.(ref_zc_corr[2, :]), ylim = (0, max_mag0), title = "NID = 1"), 
	plot(t_corr, abs.(ref_zc_corr[3, :]), ylim = (0, max_mag0), title = "NID = 2"), 
	layout = (3, 1), 
	legend = false
)
display(ref_ZadoffChuPlot)

sur_ZadoffChuPlot = plot(
	plot(t_corr, abs.(sur_zc_corr[1, :]), ylim = (0, max_mag1), title = "NID = 0"),  
	plot(t_corr, abs.(sur_zc_corr[2, :]), ylim = (0, max_mag1), title = "NID = 1"), 
	plot(t_corr, abs.(sur_zc_corr[3, :]), ylim = (0, max_mag1), title = "NID = 2"), 
	layout = (3, 1), 
	legend = false
)
display(sur_ZadoffChuPlot)


# Remove direct path from reference signal
#sur_sig, delay = align_signals(ref_sig, sur_sig)
#sur_sig = remove_direct_path(ref_sig, sur_sig)
#sur_sig = kalman_filter(sur_sig, 0.01, 0.1)


dir["IQStream"] = ref_sig
refl["IQStream"] = sur_sig

pulseIdx = 0
for pulse in 1:nPulses
    global dir["IQDataMap"][pulse, :] = dir["IQStream"][pulseIdx.+(1:curframeSamp), 1]
	global refl["IQDataMap"][pulse, :] = refl["IQStream"][pulseIdx.+(1:curframeSamp), 1]
    global pulseIdx = pulseIdx + Int(round(curframeSamp))
end



# flip streams to match processing 
dir["IQStream"] = transpose.(dir["IQStream"])
refl["IQStream"] = transpose.(refl["IQStream"])

# Create a Matched filter of the IQ data map. Use the dir IQDataMap to
# match with the refl IQDataMap
dir["MatchedMap"] = Array{Complex{Float64}}(undef, nPulses, curframeSamp)
refl["MatchedMap"] = Array{Complex{Float64}}(undef, nPulses, curframeSamp)

zc_padded = transpose.(zc_padded)

for pulse in 1:nPulses
    if PSScor # use if we want to correlate simply based off of the PSS with the direct path and the reflected path
		refl["MatchedMap"][pulse, :] = ifft(fft(refl["IQDataMap"][pulse, :]) .* conj(fft(zc_padded[1, :])))
	else
		refl["MatchedMap"][pulse, :] = ifft(fft(refl["IQDataMap"][pulse, :]) .* conj(fft(dir["IQDataMap"][pulse, :])))
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

h = plot(
	plot(xaxis, abs.(refl["IQDataMap"][1, :])), 
	plot(xaxis, abs.(dir["IQDataMap"][1, :])), 
	plot(xaxis, abs.(refl["IQDataMap"][2, :])), 
	plot(xaxis, abs.(dir["IQDataMap"][2, :])), 
	layout = (4, 1), 
	size = (800, 600)
)
display(h)

#refl["MatchedMap"] = refl["MatchedMap"][:, 10:Int(end/2)] # take off one of the samples at the beginning for accuracy 

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
if length(value) < 0
	_, idx = findmax(abs.(value))
	maxIdx = exceedances[idx, 1]
	maxVal = value[idx]
end

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

nPwr = 20 * log10(mean(abs.(RanDopMap))) + 20 # nPwr goes to power of minimum detection

color_limits = (nPwr - 50, nPwr + 50)

# Plot the surface
p = plot(
    X, Y, 20 * log10.(abs.(RanDopMap)),
    st=:surface, c=:viridis,
    xlabel="Range", ylabel="Velocity",
    zlabel="Power (dB)", colorbar=true, zlims=color_limits,
	camera=(60, 20) #(90, 90) is top view (270, 90) is top with higher range above
)
display(p)

# Set axes limits
xlims!(0, maxRange)
ylims!(dopVec[end], dopVec[1])
zlims!(color_limits)

# Convert real-world coordinates to indices
#range_indices = round.(Int, (adjustedExceedances[:, 2] .- minimum(X)) ./ step(X))
#velocity_indices = round.(Int, (adjustedExceedances[:, 1] .- minimum(Y)) ./ step(Y))

# Ensure indices are within bounds
#range_indices = clamp.(range_indices, 1, size(RanDopMap, 2))
#velocity_indices = clamp.(velocity_indices, 1, size(RanDopMap, 1))

# Extract power at exceedance points
#plotPower = [20 * log10(abs(RanDopMap[v, r])) for (v, r) in zip(velocity_indices, range_indices)]

# Highlight exceedances
#scatter3d!(
#    adjustedExceedances[:, 2], # Range values (original coordinates)
#    adjustedExceedances[:, 1], # Velocity values (original coordinates)
#    plotPower,
#    marker=:circle, color=:red
#)
