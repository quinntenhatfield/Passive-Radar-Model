# Functions

# Create a doppler shift wave and apply it to the input signal
function dopShift(signalOld, doppler, sr, tStart)
    t = tStart .+ (1/sr:1/sr:length(signalOld)/sr)
    shiftWave = exp.(1im * 2 * π * doppler .* t)
    return signalOld .* shiftWave
end

# timeShift
# Create a fractional sample delay
function timeShift(iSignal, ftd)
    # Append 1 sample to the end of iSignal (so that after we shift
    # it by a fractional sample it still fits)
    iSignal = vcat(iSignal, 0)
    
    # MCCMARKER
    N = 63
    t = collect(-(N-1)/2:(N-1)/2)
    fracDelayFilter = sin.(pi .* (t .- ftd)) ./ (π .* (t .- ftd))
    oSignal = conv(iSignal , fracDelayFilter)
	oSignal = oSignal[1:length(iSignal)] #making it 'same' like in MATLAB
	
    # end marker
end

function timeShift2(iSignal, range, fs, c)
    td = ceil(range / c * fs)
    return vcat(zeros(td), iSignal)
end

# Frank-Zadoff-Chu method
# Params:
# u: root index [25, 29, 34] corresponding to N_ID^2 [0, 1, 2],
# respectively
# Returns:
# ZC: Frank-Zadoff-Chu sequence
function zadoffChu(u)
    n = 0:62
    zc = exp.(-1im * π * u .* n .* (n .+ 1) / 63) # performs Zadoff-Chu on all numbers in array 'n'
    zc[31] = 0 # removes the DC carrier at the middle of the sequence (number 31)
    return zc
end

function zadoffChu2(u)
    n1 = 0:30
    n2 = 31:61
    zc1 = exp.(-1im * π * u * n1 .* (n1 .+ 1) / 63)
    zc2 = exp.(-1im * π * u * (n2 .+ 1) .* (n2 .+ 2) / 63)
    return vcat(zc1, 0, zc2)
end

