clc;
close all;
clear;

%% Constants of light and noise
boltz = 1.380649*10^-23;
c = 2.998e8; % speed of light
totalTime = 0;
temp = 2.9e2;

 
%% EnodeBOutput parameters
RefChannel = 2; % reference channels based off of LTE Toolbox
DuplexMode = 'FDD'; % either FDD or TDD
NCellID = 21; % any valid NCID
TotSubframes = 200;

%% Processing Options
pulseLength = 'slot'; % subframe, halfframe, frame, slot(half of a subframe)
downsamp = false; % should we downsample?
PSScor = false; % if you want to correlate w the PSS signals (must also use halfframe pulse length
antennaType = 'dir'; % either 'omni' or 'direct' for reflection antenna
randomTargets = true; % T/F whether you want randomly generated targets or manual
numTar = 10; % number of targets if 'randomTargets' is true

%% Simulation Options
simTime = 0;
writeVid = false;
loadFromFile = 0;


if loadFromFile
    
    load eNodeBOutput.mat;
    sr = 3840000; %depending on what the file was sampled at
    noiseCoef = 0;
    TotSubframes = TotSubframes;

elseif RefChannel == 2

    cfg = struct('RC', 'R.2', ...
        'DuplexMode', DuplexMode, ...
        'NCellID', NCellID, ...
        'TotSubframes', TotSubframes, ...
        'NumCodewords', 1, ...
        'Windowing', 0, ...
        'AntennaPort', 1);
    
    cfg.OCNGPDSCHEnable = 'Off';
    cfg.OCNGPDCCHEnable = 'Off';
    cfg.PDSCH.TxScheme = 'Port0';
    cfg.PDSCH.RNTI = 1;
    cfg.PDSCH.Rho = 0;
    cfg.PDSCH.RVSeq = [0 1 2 3];
    cfg.PDSCH.NHARQProcesses = 8;
    cfg.PDSCH.PMISet = 1;
    cfg = lteRMCDL(cfg);
    % starter values, can be changed below if needed
    noiseCoef = 1;


elseif RefChannel == 4

    cfg = struct('RC', 'R.4', ...
        'DuplexMode', DuplexMode, ...
        'NCellID', NCellID, ...
        'TotSubframes', TotSubframes, ...
        'NumCodewords', 1, ...
        'Windowing', 0, ...
        'AntennaPort', 1);
    
    cfg.OCNGPDSCHEnable = 'Off';
    cfg.OCNGPDCCHEnable = 'Off';
    cfg.PDSCH.TxScheme = 'Port0';
    cfg.PDSCH.RNTI = 1;
    cfg.PDSCH.Rho = 0;
    cfg.PDSCH.RVSeq = [0 1 2 3];
    cfg.PDSCH.NHARQProcesses = 8;
    cfg.PDSCH.PMISet = 1;
    cfg = lteRMCDL(cfg);   
    % starter values, can be changed below if needed
    noiseCoef = 3;

elseif RefChannel == 5

    cfg = struct('RC', 'R.5', ...
        'DuplexMode', DuplexMode, ...
        'NCellID', NCellID, ...
        'TotSubframes', TotSubframes, ...
        'NumCodewords', 1, ...
        'Windowing', 0, ...
        'AntennaPort', 1);
    
    cfg.OCNGPDSCHEnable = 'Off';
    cfg.OCNGPDCCHEnable = 'Off';
    cfg.PDSCH.TxScheme = 'Port0';
    cfg.PDSCH.RNTI = 1;
    cfg.PDSCH.Rho = 0;
    cfg.PDSCH.RVSeq = [0 1 2 3];
    cfg.PDSCH.NHARQProcesses = 8;
    cfg.PDSCH.PMISet = 1;
    cfg = lteRMCDL(cfg); 
    % starter values, can be changed below if needed
    noiseCoef = 4;
    
elseif RefChannel == 9

    cfg = struct('RC', 'R.9', ...
        'DuplexMode', DuplexMode, ...
        'NCellID', NCellID, ...
        'TotSubframes', TotSubframes, ...
        'NumCodewords', 1, ...
        'Windowing', 0, ...
        'AntennaPort', 1);
    
    cfg.OCNGPDSCHEnable = 'Off';
    cfg.OCNGPDCCHEnable = 'Off';
    cfg.PDSCH.TxScheme = 'Port0';
    cfg.PDSCH.RNTI = 1;
    cfg.PDSCH.Rho = 0;
    cfg.PDSCH.RVSeq = [0 1 2 3];
    cfg.PDSCH.NHARQProcesses = 8;
    cfg.PDSCH.PMISet = 1;
    cfg = lteRMCDL(cfg);
    noiseCoef = 1;
end

% assign the signal to 'eNodeBOutput'
if ~loadFromFile
    % input bit source:
    in = [1; 0; 0; 1];
    [eNodeBOutput, grid, cfg] = lteRMCDLTool(cfg, in); 
    sr = cfg.SamplingRate;
end


%add noise to the signal (as would normally happen), but keep clean
%signal for transmission purposes. More just to see ZC correlation with noise
noiseStream = (randn(size(eNodeBOutput))+1i*randn(size(eNodeBOutput))*sqrt(boltz*temp*sr/2)) * noiseCoef;
eNodeBOutputNoisy = eNodeBOutput + noiseStream;


% plot clean and noisy signal for reference
figure
subplot(2, 1, 1)
plot(abs(eNodeBOutput));
title('Clean Signal')
subplot(2, 1, 2)
plot(abs(eNodeBOutputNoisy));
title('Noisy Signal')

% plot the clean and noisy signals in the frequency domain
figure
subplot(2, 1, 1)
plot(abs(fftshift(fft(eNodeBOutput))));
title('Clean Signal')
subplot(2, 1, 2)
plot(abs(fftshift(fft(eNodeBOutputNoisy))));
title('Noisy Signal')


%% Noise Control
% if not selected for one of the presets
% noiseCoef = 4;


%% Direct Path Receiver Control
dir.freq = 700e6; % antenna frequency (Hz)
dir.antGain = 5; % antenna dBi (assumes same tx/rx)
dir.power = 40e3; % Tx Power (W)
dir.temp = 2.9e2; % Noise Power controlled by temp (K) (also controlled by sr)
dir.noiseCells = 50; % For noise estimation

%% Reflected Path Receiver Control
refl.freq = 700e6; % antenna frequency (Hz)
refl.antGain = 5; % antenna dBi (assumes same tx/rx)
refl.power = 40e3; % Tx Power (W)
refl.temp = 2.9e2; % Noise Power controlled by temp (K) (also controlled by sr)
refl.noiseCells = 50; % For noise estimation
refl.antType = antennaType;

%% Create Transmitter and Receiver (Static)
% Transmitter
txPos = [-1000 -1000 100];
txVel = [0 0 0];
% Receiver
rxPos = [0 0 0];
rxVel = [0 0 0];

dirPath.range = norm(txPos - rxPos);
dirPath.velocity = norm(txVel - rxVel);

%% Target Creation
targets = struct(); % Target Definition

if randomTargets % target creation for loop for random targets   
    for tar = 1:numTar
    posRange = 5000;
    velRange = 25;
    accRange = rand();
    targets(tar).position = [rand() rand() rand()] * posRange;
    targets(tar).RCS = 20;
    targets(tar).velocity = [randn() randn() randn()] * velRange;
    targets(tar).acceleration = [randn() randn() randn()] * accRange;
    end
else    % 4 targets written in manually
    targets(1).position = [0 5000 5000];
    targets(1).RCS = 20;
    targets(1).velocity = [0 0 0];
    targets(1).acceleration = [0 0 0];
    
    targets(2).position = [5000 0 4000];
    targets(2).RCS = 20;
    targets(2).velocity = [4 50 50];
    targets(2).acceleration = [0 0 0];
    
    targets(3).position = [1000 1000 5000];
    targets(3).RCS = 20;
    targets(3).velocity = [4 25 25];
    targets(3).acceleration = [0 0 0];
    
    targets(4).position = [2000 3000 4000];
    targets(4).RCS = 20;
    targets(4).velocity = [13 0 12];
    targets(4).acceleration = [0 0 0];
end

%% Find Relative Positioning and Velocity of Targets
for tar = 1:length(targets)
    targets(tar).rangeTx = norm(targets(tar).position-txPos); 
    targets(tar).rangeRx = norm(targets(tar).position-rxPos);
    targets(tar).rangeTotal = targets(tar).rangeTx + targets(tar).rangeRx; % range from Tx->Tar->Rx
    dt = targets(tar).position - txPos; % distance in vector form
    dr = targets(tar).position - rxPos;
    vt = targets(tar).velocity - txVel; % velocity in vector form
    vr = targets(tar).velocity - rxVel;
    targets(tar).velocityTx = dot(dt,vt)/norm(dt); % velocity with respect to Tx/Rx
    targets(tar).velocityRx = dot(dr,vr)/norm(dr); 
end

% %% Plot IQ signals and FFT
% figure %#ok<UNRCH> 
% hold on
% %t = (1/dsr:1/dsr:pw) - pw/2;
% plot(real(eNodeBOutput))
% plot(imag(eNodeBOutput))
% hold off
% figure
% plot(((-length(eNodeBOutput)+1)/2:(length(eNodeBOutput)-1)/2)*sr/length(eNodeBOutput),abs(fftshift(fft(eNodeBOutput))))


%% Option to downsample eNodeBOutput to just the 6 RB of the PSS/SSS
if downsamp
    enb = struct;
    enb.NDLRB = 6; % number of DL resource blocks
    ofdmInfo = lteOFDMInfo(setfield(enb,'CyclicPrefix','Normal')); %#ok<SFLD>
    if (sr~=ofdmInfo.SamplingRate)
        if (sr < ofdmInfo.SamplingRate)
            warning('The received signal sampling rate (%0.3fMs/s) is lower than the desired sampling rate for cell search / MIB decoding (%0.3fMs/s); cell search / MIB decoding may fail.',sr/1e6,ofdmInfo.SamplingRate/1e6);
        end
        fprintf('\nResampling from %0.3fMs/s to %0.3fMs/s for cell search / MIB decoding...\n',sr/1e6,ofdmInfo.SamplingRate/1e6);
    else
        fprintf('\nResampling not required; received signal is at desired sampling rate for cell search / MIB decoding (%0.3fMs/s).\n',sr/1e6);
    end
    % Downsample received signal
    nSamples = ceil(ofdmInfo.SamplingRate/round(sr)*size(eNodeBOutput,1));
    nRxAnts = size(eNodeBOutput, 2);
    downsampled = zeros(nSamples, nRxAnts);
    for i=1:nRxAnts
        downsampled(:,i) = resample(eNodeBOutput(:,i), ofdmInfo.SamplingRate, round(sr));
    end
    noiseCoef = noiseCoef/2;
    eNodeBOutput = downsampled;
    sr = ofdmInfo.SamplingRate;
end

%% Creating PRF info
% each length includes pulsewidht, number of pulses in the amount of
% subframes we sampled, the time of that type of that pulse (pw which is
% redundant), amount of samples for curr sr, and the first section in the signal 
if strcmp(pulseLength, 'subframe')
    pw = 1e-3;
    nPulses = TotSubframes; 
    curframeTime = .001;
    curframeSamp = curframeTime * sr;
    curframe0 = eNodeBOutput(1:curframeSamp, 1); % clean version of the 1st subframe
%     curframe0 = eNodeBOutput(5*curframeSamp+1:6*curframeSamp, 1); % clean
%     version of the 6th subframe block (which is the 2nd PSS/SSS in a frame)
    curframe0Noisy = eNodeBOutputNoisy(1:curframeSamp, 1);% noise

elseif strcmp(pulseLength, 'halfframe')
    pw = 5e-3;
    nPulses = TotSubframes/5;
    curframeTime = .005;
    curframeSamp = curframeTime * sr;
    curframe0 = eNodeBOutput(1:curframeSamp, 1);% clean
    curframe0Noisy = eNodeBOutputNoisy(1:curframeSamp, 1);% noise

elseif strcmp(pulseLength, 'frame')
    pw = 10e-3;
    nPulses = TotSubframes/10; 
    curframeTime = .01;
    curframeSamp = curframeTime * sr;
    curframe0 = eNodeBOutput(1:curframeSamp, 1);% clean
    curframe0Noisy = eNodeBOutputNoisy(1:curframeSamp, 1);% noise

elseif strcmp(pulseLength, 'slot')
    pw = .5e-3;
    nPulses = TotSubframes; 
    curframeTime = .0005;
    curframeSamp = curframeTime * sr;
    curframe0 = eNodeBOutput(1:curframeSamp, 1);   
    curframe0Noisy = eNodeBOutputNoisy(1:curframeSamp, 1);% noise

else
    disp("Set Desired Pulselength");
end

% update PRF stuff
prfSpacing = 0;
prfStart = 1/pw;
prf = prfStart*ones(1,nPulses);
prf = prf(randperm(length(prf)));
% not dealing with fractional samples, so we adjust the PRI to the nearest samp
for count = 1:nPulses
    prf = 1./(round(sr./prf)/sr);
    assert(rem(sr,prf(count))==0)
end

%% Reading LTE For PSS
% all basic timing for 20MHz at sampling rate sr_basic or 30.72 Msps 
sr_basic = 30720000;
symbTime_basic = 2048; % time of a symbol (samples)
cp_0_basic = 160; % time of cyclic prefix of the 0th symbol
cp_1_basic = 144; % time of cyclic prefix of symbols 1-6 
%conversion of symbol timing in sampled
symbTime = symbTime_basic / sr_basic * sr;
cp_0 = cp_0_basic / sr_basic * sr;
cp_1 = cp_1_basic / sr_basic * sr;

%delta = 0; % phase rotation TO BE MEASURED LATER?

% time of clean signal
time = length(eNodeBOutput) / sr; %seconds

subcarriers_ss = floor(symbTime/2) - 3 * 12 + 5 : floor(symbTime/2) + 3 * 12 - 5;   % gives us the 62 subcarriers of both SS, and the DC in the middle
% above could be written - 31 and + 31

% create the 3 time domain ZC sequences corresponding to 0,1,2 NCID2s
zc_u = [25, 29, 34]; % u values in the Zadoff-Chu sequence. 
zc_time = zeros(3, symbTime);
zc_all = zeros(3, symbTime);
for u = 1:3
    zc = zadoffChu(zc_u(u));
    zc_all = zeros(1, symbTime);
    zc_all(u, subcarriers_ss) = zc;
    zc_time(u, :) = ifft(fftshift(zc_all(u, :)));%
end

% Matched filter with ZC to find NCID2
zc_corr = ifft(fft(curframe0, size(curframe0, 1),1).*conj(fft(zc_time.', size(curframe0, 1))), size(curframe0, 1), 1);
zc_corr = zc_corr.';

% Find NCID2 from peak corr and find sample and time offset
[corrTimingPeakY, corrTimingPeakSamp] = max(zc_corr, [], 2);
[absPeak, NCID2]  = max(corrTimingPeakY);
absPeakAbs = abs(absPeak);
pssStart = corrTimingPeakSamp(NCID2) / sr;
%NCID2 = NCID2 - 1; %technically the NCID  values are 0-2 instead of 1-3
%but not useful if using NCID2 later.. just keep it in mind

t_corr = 0:1/sr:curframeTime - 1/sr; % time of correlation for x axis values

% first frame (any length) correlated with the 3 zadoff-chu sequences
figure(70)
subplot(3, 1, 1)
plot(t_corr, abs(zc_corr(1, :)));
ylim([0 absPeakAbs])

subplot(3, 1, 2)
plot(t_corr, abs(zc_corr(2, :)));
ylim([0 absPeakAbs])

subplot(3, 1, 3)
plot(t_corr, abs(zc_corr(3, :)));
ylim([0 absPeakAbs])


% clean pulse map to be used for transmission
pulseIdx = 0;
frameMap = zeros(nPulses, curframeSamp);
for pulse = 1:nPulses
    frameMap(pulse, :) = eNodeBOutput(pulseIdx+(1:size(frameMap, 2)), 1);
    pulseIdx = pulseIdx + round(curframeSamp);
end


% video frame counter
m = 1;
%clock to run live sim
tic;
while totalTime <= simTime

    %% Send LTE and recieve direct path with stationary Tx and Rx
    
    % Create the signal as an IQ Stream to match how each receiver may receive it
    dir.IQStream = (randn(length(eNodeBOutput),1)+1i*randn(length(eNodeBOutput),1))*sqrt(boltz*dir.temp*sr/2)*noiseCoef;
    refl.IQStream = (randn(length(eNodeBOutput),1)+1i*randn(length(eNodeBOutput),1))*sqrt(boltz*dir.temp*sr/2)*noiseCoef;
    
    % Generate the sizes of the IQ Data Maps 
    dir.IQDataMap = (zeros(nPulses,round(curframeSamp))+1i*zeros(nPulses,round(curframeSamp)));    
    refl.IQDataMap = (zeros(nPulses,round(curframeSamp))+1i*zeros(nPulses,round(curframeSamp)));
    
    
    % direct path 
    eNodeBOutput = eNodeBOutput.';
    pw = curframeTime;
    pulseIdx = 0;
    for pulse = 1:nPulses
        dirPath.pScale = 10^((log10(dir.power) + 2*dir.antGain + 20*log10(c/dir.freq) - 30*log10(4*pi) - 20*log10(dirPath.range))/20);
        dirPath.doppler = -dirPath.velocity * dir.freq / c;
        pStart = floor(dirPath.range / c * sr);
        pEnd = min(floor(((dirPath.range/c) + pw) * sr), length(dir.IQStream));
        if pEnd + pulseIdx > length(dir.IQStream) % helps account for overflow of the last LTE pulse
            pEnd = length(dir.IQStream) - pulseIdx;
        end  
        %changed pEnd to go to the whole signal
        if pStart < 1
            continue;
        end
        pFrac = rem(dirPath.range / c * sr, 1);
        tResponse = dopShift(frameMap(pulse, :), dirPath.doppler, sr, sum(1./prf(1:pulse-1))) * dirPath.pScale * exp(-1i * 2 * pi * pFrac);
        tResponse = timeShift(tResponse, rem(dirPath.range * sr / c, 1));
        dir.IQStream(pulseIdx + (pStart : pEnd)) = dir.IQStream (pulseIdx + (pStart:pEnd)) + tResponse(1:pEnd - pStart + 1).';
        if strcmp(refl.antType, 'omni')
            refl.IQStream(pulseIdx + (pStart : pEnd)) = refl.IQStream (pulseIdx + (pStart:pEnd)) + tResponse(1:pEnd - pStart + 1).';
        end
        pulseIdx = pulseIdx + round(sr/prf(pulse));
        txPos = txPos + txVel/prf(pulse);
        rxPos = rxPos + rxVel/prf(pulse);
        dirPath.range = norm(txPos - rxPos);
    end    
    
    
    
    for tar = 1:length(targets)
        
        pulseIdx = 0;
        for pulse = 1:nPulses
            targets(tar).pScale = 10^((10*log10(refl.power) + 2*refl.antGain + targets(tar).RCS...
                + 20*log10(2.998e8/refl.freq) - 30*log10(4*pi) - 20*log10(targets(tar).rangeTx)...
                - 20*log10(targets(tar).rangeRx))/20);
            % Change doppler to frquency shift
            targets(tar).doppler = -(targets(tar).velocityTx+targets(tar).velocityRx)*refl.freq/2.998e8;
            % Find the start and stop index for where to insert the pulse
            pStart = floor((targets(tar).rangeTx+targets(tar).rangeRx) / 2.998e8 * sr);
            pEnd = min(floor((((targets(tar).rangeTx+targets(tar).rangeRx) / 2.998e8) + pw) * sr),length(refl.IQStream));
            if pEnd + pulseIdx > length(refl.IQStream) % helps account for overflow of the last LTE pulse
                pEnd = length(refl.IQStream) - pulseIdx;
            end       
            % pStart should never be less than zero/one
            if pStart < 1
                continue;
            end
            pFrac = rem((targets(tar).rangeTx+targets(tar).rangeRx) / 2.998e8 * sr,1);
            % Apply doppler shift and phase shift to clean pulse
            tResponse = dopShift(frameMap(pulse, :),targets(tar).doppler,sr,sum(1./prf(1:pulse-1)))*targets(tar).pScale*exp(-1i*2*pi*pFrac); %
            tResponse = timeShift(tResponse,rem((targets(tar).rangeTx+targets(tar).rangeRx) * sr / 2.998e8 , 1));
            
            % Add the modified pulse to the IQ Data Map
            refl.IQStream(pulseIdx+(pStart:pEnd)) = refl.IQStream(pulseIdx+(pStart:pEnd)) + tResponse(1:pEnd-pStart+1).';
            pulseIdx = pulseIdx + round(sr/prf(pulse));
    
            % Kinematics updates
            txPos = txPos + txVel/prf(pulse);
            rxPos = rxPos + rxVel/prf(pulse);
            targets(tar).position = targets(tar).position + targets(tar).velocity/prf(pulse);
            targets(tar).velocity = targets(tar).velocity + targets(tar).acceleration/prf(pulse);
    
            % Recalculate range and vel (relative)
            targets(tar).rangeTx = norm(targets(tar).position-txPos);
            targets(tar).rangeRx = norm(targets(tar).position-rxPos);
            targets(tar).rangeTotal = targets(tar).rangeTx + targets(tar).rangeRx;
            dt = targets(tar).position - txPos;
            dr = targets(tar).position - rxPos;
            vt = targets(tar).velocity - txVel;
            vr = targets(tar).velocity - rxVel;
            targets(tar).velocityTx = dot(dt,vt)/norm(dt);
            targets(tar).velocityRx = dot(dr,vr)/norm(dr);
    
        end
    
    end
    
    
    % Subtract the direct IQStream from the reflected IQStream if omni.
    % May have to multiply direct IQStream by a coefficient to match the power of
    % the pulse being removed if they are uneven due to antenna gain
    if strcmp(refl.antType, 'omni')
        refl.IQStream = refl.IQStream - dir.IQStream;
    end
    
    % Pull the IQStream data and place it appropriately in the IQDataMap
    pulseIdx = 0;
    % eNodeBOutput = eNodeBOutput.';
    for pulse = 1:nPulses
        dir.IQDataMap(pulse,:) = dir.IQStream(pulseIdx+(1:size(dir.IQDataMap,2)),1);
        refl.IQDataMap(pulse,:) = refl.IQStream(pulseIdx+(1:size(refl.IQDataMap,2)), 1); 
        pulseIdx = pulseIdx + round(curframeSamp);
    end
    
%     % Plot of direct IQDataMap (1, 4)
%     figure(20)
%     subplot(4, 1, 1)
%     plot(abs(dir.IQDataMap(1, :)));
%     subplot(4, 1, 2)
%     plot(abs(dir.IQDataMap(2, :)));
%     subplot(4, 1, 3)
%     plot(abs(dir.IQDataMap(3, :)));
%     subplot(4, 1, 4)
%     plot(abs(dir.IQDataMap(4, :)));
%     
%     % Plot of reflection IQDataMap (1, 4)
%     figure(21)
%     subplot(4, 1, 1)
%     plot(abs(refl.IQDataMap(1, :)));
%     subplot(4, 1, 2)
%     plot(abs(refl.IQDataMap(2, :)));
%     subplot(4, 1, 3)
%     plot(abs(refl.IQDataMap(3, :)));
%     subplot(4, 1, 4)
%     plot(abs(refl.IQDataMap(4, :)));
    
    
    % flip streams to match processing 
    dir.IQStream = dir.IQStream.';
    refl.IQStream = refl.IQStream.';       
    
    % Create a Matched filter of the IQ data map. Use the dir IQDataMap to
    % match with the refl IQDataMap
    dir.MatchedMap = zeros(nPulses, curframeSamp);
    refl.MatchedMap = zeros(nPulses, curframeSamp);
    for pulse = 1:nPulses
        if PSScor % use if we want to correlate simply based off of the PSS with the direct path and the reflected path
            dir.MatchedMap(pulse, :) = ifft(fft(dir.IQDataMap(pulse, :), size(dir.IQDataMap, 2), 2).*conj(fft(zc_time(NCID2, :), size(dir.IQDataMap, 2))), size(dir.IQDataMap, 2), 2);            
            refl.MatchedMap(pulse, :) = ifft(fft(refl.IQDataMap(pulse, :), size(refl.IQDataMap, 2), 2).*conj(fft(zc_time(NCID2, :), size(refl.IQDataMap, 2))), size(refl.IQDataMap, 2), 2); 
            refl.MatchedMap(pulse, :) = ifft(fft(refl.MatchedMap(pulse, :), size(refl.MatchedMap, 2), 2).*conj(fft(dir.MatchedMap(pulse, :), size(refl.IQDataMap, 2))), size(refl.IQDataMap, 2), 2);             
        else
            refl.MatchedMap(pulse, :) = ifft(fft(refl.IQDataMap(pulse, :), size(refl.IQDataMap, 2), 2).*conj(fft(dir.IQDataMap(pulse, :), size(refl.IQDataMap, 2))), size(refl.IQDataMap, 2), 2);   
        end
    end
    
    
%     figure(60)
%     subplot(4, 1, 1)
%     plot(abs(dir.MatchedMap(1, :)));
%     subplot(4, 1, 2)
%     plot(abs(dir.MatchedMap(2, :)));
%     subplot(4, 1, 3)
%     plot(abs(dir.MatchedMap(3, :)));
%     subplot(4, 1, 4)
%     plot(abs(dir.MatchedMap(4, :)));

    figure(61)
    subplot(4, 1, 1)
    plot(abs(refl.MatchedMap(1, :)));
    subplot(4, 1, 2)
    plot(abs(refl.MatchedMap(2, :)));
    subplot(4, 1, 3)
    plot(abs(refl.MatchedMap(3, :)));
    subplot(4, 1, 4)
    plot(abs(refl.MatchedMap(4, :)));
    
    refl.MatchedMap = refl.MatchedMap(:, 2:end); % take off one of the samples at the beginning for accuracy 
    
    %% CPI Control
    bw = sr / 1.536;% Hz max for any LTE signal at a specific sr
    
    %% Target Detection
    pulseBlanking = true;
    coherentProcessing = true;
    runMTI = false;
    interpFactor = 1;
    maxRange = 20000;
    %maxRange = 2.998e8/max(prf);
    cellRes = ceil(sr/bw); 
    
    constThresh = -95;
    drawExceed = false; % drawExceed uses the const threshold but everything else is CFAR
    pfa = 1e-6; % CFAR Parameters
    nGuardLead = cellRes*10;
    nLag = 10;
    nGuardLag = cellRes*10;
    nLead = 10;
    
    x = [0 (cumsum(1./(prf(1:end-1))))]; %#ok<UNRCH>
    nPulses = nPulses*interpFactor;
    xq = linspace(0,x(end),nPulses);
    % Optional Windowing
    window = ones(1,length(xq));%hamming(length(xq)).';
    windowLoss = -20*log10(mean(window));
    newPRF = 1/xq(2);
    dir.InterpMap = zeros(size(dir.MatchedMap));
    refl.InterpMap = zeros(size(refl.MatchedMap));
    for idx = 1:size(refl.MatchedMap,2)
        v = refl.MatchedMap(:,idx);
        % Interpolate to allow for doppler processing with staggered PRIs
        refl.InterpMap(:,idx) = interp1(x,v,xq,'linear').*window;%nearest
    
        v = refl.MatchedMap(:,idx);
        refl.InterpMap(:,idx) = interp1(x,v,xq,'linear').*window;%nearest
        
    end
    
    RanDopMap = fftshift(fft(refl.InterpMap,length(xq),1),1);

    %% CFAR
        % Run CFAR to get detections
        [exceedances, thresholdMap] = cfar(RanDopMap, pfa, ...
                               nLead, nGuardLead, ...
                               nLag, nGuardLag, ...
                               false,false);
    
        simpleExceed = exceedances;
    
        % Calculate range and doppler vectors for the plots
        rngVec = linspace(0, maxRange, size(RanDopMap,2)); % return rdMap compatible range vector (meters)
        dopVec = linspace(-2.998e8/refl.freq*newPRF/nPulses*(-nPulses)/2, -2.998e8/refl.freq*newPRF/nPulses*(nPulses-2)/2, nPulses);       % return rdMap compatible doppler vector (Hz)
        coherentProcessing = true;
        % Create clusters for groups of exceedances
        if coherentProcessing
            clusters = combineExceedances(exceedances, cellRes*2, ceil(nPulses/4));
        else
            clusters = combineExceedances(exceedances, cellRes*2, 0);
        end
        exceedances = zeros(length(clusters),2);
        for count = 1:length(clusters)
            tempVal = zeros(size(clusters{count},1),1);
            for counter = 1:size(clusters{count},1)
                tempVal(counter) = RanDopMap(clusters{count}(counter,2),clusters{count}(counter,1));
            end
            [~,maxIdx] = max(abs(tempVal));
            exceedances(count,1) = clusters{count}(maxIdx,1);
            exceedances(count,2) = clusters{count}(maxIdx,2);
        end
        if ~coherentProcessing
            clusters = combineExceedances(exceedances, cellRes*2, nPulses);
            exceedances = zeros(length(clusters),2);
            newCount = 1;
            for count = 1:length(clusters)
                if size(clusters{count},1) < nPulses/2
                    continue;
                end
                tempVal = zeros(size(clusters{count},1),1);
                for counter = 1:size(clusters{count},1)
                    tempVal(counter) = RanDopMap(clusters{count}(counter,2),clusters{count}(counter,1));
                end
                [~,maxIdx] = max(abs(tempVal));
                exceedances(newCount,1) = clusters{count}(maxIdx,1);
                exceedances(newCount,2) = clusters{count}(maxIdx,2);
                newCount = newCount + 1;
            end
            exceedances = exceedances(1:newCount-1,:);
        end
    
        % Get the power of each exceedance
        value = zeros(1,size(exceedances,1));
        for count = 1:size(exceedances,1)
            value(count) = RanDopMap(exceedances(count,2),exceedances(count,1));
        end
        
        % Get the max power exceedance from each cluster
        [~,idx] = max(abs(value));
        maxIdx = exceedances(idx,1);
        maxVal = value(idx);
        
        clc;
        % Prepare constants to do interpolation between cells
        rngCell = rngVec(2)-rngVec(1);
        binDiff = abs(mean(diff(dopVec)));
        % Sort exceedances by range
        [~,sortExceed] = sort(exceedances(:,1),'descend');
    
        adjustedExceedances = zeros(size(exceedances));
    
        disp('Detections')
        for count = sortExceed.'
            % Get total power of relevant doppler cells
            dopInterp = abs(RanDopMap(exceedances(count,2),exceedances(count,1)));
            if exceedances(count,2) == nPulses || (exceedances(count,2) > 1 && abs(RanDopMap(exceedances(count,2)-1,exceedances(count,1))) > abs(RanDopMap(exceedances(count,2)+1,exceedances(count,1))))
                dopInterp = dopInterp + abs(RanDopMap(exceedances(count,2)-1,exceedances(count,1)));
                idxArr = [exceedances(count,2)-1 exceedances(count,2)];
            else
                dopInterp = dopInterp + abs(RanDopMap(exceedances(count,2)+1,exceedances(count,1)));
                idxArr = [exceedances(count,2) exceedances(count,2)+1];
            end
            % Interpolate with nearest neighbor
            dopArr = 0;
            for idx = idxArr
                dopArr = dopArr + dopVec(idx)*abs(RanDopMap(idx,exceedances(count,1)))/dopInterp;
            end
            % Adjust the velocity (and range?) to match the interp value
            adjustedExceedances(count,1) = exceedances(count,1)*2.998e8/sr;
            adjustedExceedances(count,2) = dopArr;
            % Print the detection data
            disp(['Range: ' num2str(exceedances(count,1)*2.998e8/sr + dirPath.range,5) 'm' char(9) 'Power: ' num2str(20*log10(abs(value(count))),4) ' dBW ' char(9) 'Velocity: ' num2str(dopArr) ' m/s'])
        end
        % Sort the truth data by range
        [~,sortTar] = sort(vertcat(targets.rangeTotal),'descend');
        disp('Targets')
        for count = sortTar.'
            % Calculate loss from between doppler cells
            [binOff,binIdx] = min(abs(dopVec-(mod((targets(count).velocityTx + targets(count).velocityRx)-dopVec(1),dopVec(1)-dopVec(end)+binDiff)-dopVec(1))));
            dopLoss = 10*log10(binDiff/(binDiff-binOff));
            % Calculate the expected power return (validates our equation)
            pReturn = 10*log10(refl.power) + 2*refl.antGain + targets(count).RCS + 20*log10(2.998e8/refl.freq) - 30*log10(4*pi) - 20*log10(targets(count).rangeTx) - 20*log10(targets(count).rangeRx) + 20*log10(nPulses) + 20*log10(sr*pw) - dopLoss - windowLoss;
            % Display the truth data
            disp(['Range: ' num2str(targets(count).rangeRx + targets(count).rangeTx,5) 'm' char(9) 'Power: ' num2str(pReturn,4) ' dBW ' char(9) 'Velocity: ' num2str(targets(count).velocityTx + targets(count).velocityRx) ' m/s'])
        end
    
        disp('Direct')
        disp(['Range: ' num2str(dirPath.range) 'm' char(9)]) %  'Velocity: ' num2str(dirPath.doppler) ' m/s'])
    
        
        % Display the range doppler map
        figure(124)
        
        [X,Y] = meshgrid((1:size(RanDopMap,2))*2.998e8/sr,-2.998e8/refl.freq*newPRF/nPulses*((-nPulses)/2:(nPulses-2)/2));
        surf(X,Y,20*log10(abs(RanDopMap)));
        shading flat;
        colorbar;
        if isempty(value)
            nPwr = -200;
        else
            nPwr = 20*log10(min(abs(value))); %nPwr goes to power of minimum detection
        end
        clim([nPwr-50,nPwr+50])
    %     xlim([0 maxRange-pw*2.998e8])
        xlim([0 maxRange])
        ylim([dopVec(end) dopVec(1)])%ylim(2.998e8/refl.freq/2*newPRF/nPulses*[(-nPulses)/2 (nPulses-2)/2])
        zlim([nPwr-50,nPwr+50])
        ylabel("Velocity")
        xlabel("Range")
        view(-60,20);
        hold on
        plotPower = diag(20*log10(abs(RanDopMap(exceedances(:,2),exceedances(:,1)))));
        plot3(adjustedExceedances(:,1), adjustedExceedances(:,2), plotPower,'rs');
        hold off
        
        M(m) = getframe(gcf);
        m = m + 1;

    % Draw the exceedance plot using constThresh. Mostly for troubleshooting
    if drawExceed
    
        exceedPlot = RanDopMap;
        exceedPlot(20*log10(abs(RanDopMap)) > constThresh) = 1;
        exceedPlot(20*log10(abs(RanDopMap)) <= constThresh) = 0;
        
        figure
        surf(X,Y,exceedPlot)
       % xlim([pw*2.998e8/2 1/newPRF*2.998e8/2-pw*2.998e8/2])
       % ylim(2.998e8/refl.freq/2*newPRF/nPulses*[(-nPulses)/2 (nPulses-2)/2])
        zlim([-1 2])
        ylabel("Velocity")
        xlabel("Range")
        view(3)
    
        figure
        surf(X,Y,thresholdMap)
        %xlim([pw*2.998e8/2 1/newPRF*2.998e8/2-pw*2.998e8/2])
       % ylim(2.998e8/refl.freq/2*newPRF/nPulses*[(-nPulses)/2 (nPulses-2)/2])
        zlim([-1 2])
        ylabel("Velocity")
        xlabel("Range")
        view(3)
    end

    
    % time keeping for subsequent runs
    drawnow;
    temptime = toc;
    tic;
    tempOfsampleRet = -cumsum(1./prf);
    totalTime = totalTime + temptime;

    % Apply kinematics
    for tar = 1:length(targets)
        txPos = txPos + txVel/prf(pulse);
        rxPos = rxPos + rxVel/prf(pulse);
        targets(tar).position = targets(tar).position + targets(tar).velocity*(temptime+tempOfsampleRet(end));
        targets(tar).velocity = targets(tar).velocity + targets(tar).acceleration*(temptime+tempOfsampleRet(end));

        targets(tar).rangeTx = norm(targets(tar).position-txPos);
        targets(tar).rangeRx = norm(targets(tar).position-rxPos);
        targets(tar).rangeTotal = targets(tar).rangeTx + targets(tar).rangeRx;
        dt = targets(tar).position - txPos;
        dr = targets(tar).position - rxPos;
        vt = targets(tar).velocity - txVel;
        vr = targets(tar).velocity - rxVel;
        targets(tar).velocityTx = dot(dt,vt)/norm(dt);
        targets(tar).velocityRx = dot(dr,vr)/norm(dr);
    end
    % Undo the interp factor to clean up if we use it
    if coherentProcessing
        nPulses = nPulses/interpFactor;
    end
end

% write the video and save to .avi file
if (writeVid)
    V = VideoWriter('randop.avi');
    open(V)
    for k = 1:length(M)
        writeVideo(V, M(k));
    end
    close(V)
end




%% Functions

% Create a doppler shift wave and apply it to the input signal
function signalNew = dopShift(signalOld,doppler,sr,tStart)
    t = tStart + (1/sr:1/sr:length(signalOld)/sr);
    shiftWave = exp(1i*2*pi*doppler*t);
    signalNew = signalOld.*shiftWave;
end

% timeShift
% Create a fractional sample delay
function oSignal = timeShift(iSignal, ftd)
    % Append 1 sample to the end of iSignal (so that after we shift
    % it by a fractional sample it still fits)
    iSignal = [ iSignal 0];
    %{    
    % Convert to frequency domain
    iSignal_f = fft(iSignal);

    N = length(iSignal_f);
    
    k = 0 : 1 : N-1;

    % Apply a phase ramp to time-shift
    oSignal_f = iSignal_f .* exp( -1 * 1i * 2 * pi * k * ftd / N );

    % Convert back to time domain
    oSignal = ifft(oSignal_f);
    %}
    
    %MCCMARKER
    N = 63;
    t = -(N-1)/2:(N-1)/2;
    fracDelayFilter = sin(pi*(t-ftd))./(pi*(t-ftd));
    oSignal = conv(iSignal,fracDelayFilter,'same');
    %end marker
end

function oSig = timeShift2(iSignal, range, fs, c)
    td = ceil(range / c * fs);
    oSig = [zeros(td,1); iSignal];
end

% Frank-Zadoff-Chu method
% Params:
% u: root index [25, 29, 34] corresponding to N_ID^2 [0, 1, 2],
% respectively
% Returns:
% ZC: Frank-Zadoff-Chu sequence
function zc = zadoffChu(u)
    n = 0:62;
    zc = exp(-1i * pi * u .* n .* (n + 1) / 63);        % performs Zadoff-Chu on all numbers in array 'n'
    zc(31) = 0;        % removes the DC carrier at the middle of the sequence (number 31)
end

function zc = zadoffChu2(u)
    n1 = 0:30;
    n2 = 31:61;
    zc1 = exp(-1j * pi * u * n1 .* (n1 + 1) / 63);
    zc2 = exp(-1j * pi * u * (n2 + 1) .* (n2 + 2) / 63);
    zc = [zc1 0 zc2];
end


