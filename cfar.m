% CA-CFAR (1D)
%
% Perform one-dimensional cell-averaging constant false alarm rate
% (CA-CFAR 1D) processing on a range-Doppler map to detect targets
%
% We perform the CFAR across range bins for each Doppler
% filter. This algorithm assumes that the statistics of the
% interference are stable in a small range region around the cell
% under test (but does not assume that about Doppler and hence does
% not use adjacent Doppler bins).  This is a reasonable assumption
% in the presence of e.g. clutter returns
% 
% Input
%    rdMap : [ N x M ] of complex voltages
%        where N := number of Doppler bins
%        where M := number of range bins
%    pfa : desired probability of false alarm (e.g. 10^-6)
%    nLead : number of lead bins to use in estimate of background
%    nGuardLead : number of lead bins near bin under test to avoid
%    nLag : number of lag bins to use in estimate of background
%    nGuardLag : number of lag bins near bin under test to avoid
%
% Output
%    detections : [ N x 2 ] of (range bin, doppler bin) pairs
%        indicating threshold crossings (i.e. detections)
%
% Example Diagram
% nLead = 3, nGuardLead = 1, nLag = 3, nGuardLag = 1
%
% 1 2 3 4 5 6 7 8 9
% L L L G X G L L L
%
% Bins 1 through 3 are the "lead" bins
% Bin 4 is the "lead guard" bin
% Bin 5 is the bin under test
% Bin 6 is the "lag guard" bin
% Bins 7 through 9 are the "lag" bins
%
% References:
% [1] Richards, M.  "Fundamentals of Radar Signal Processing"
function [detections, thresholdMap] = cfar(rdMap, pfa, ...
                           nLead, nGuardLead, ...
                           nLag, nGuardLag, ...
                           debug,useMitre)

nBin = nLead + nLag;

alphaScale = computeAlphaScale(pfa, nBin);

nDopBin = size(rdMap, 1);
nRngBin = size(rdMap, 2);

if useMitre
% Presize a map for the threshold
%    We'll set the values to infinity by default.  As we loop
%    through the range and Doppler bins, we'll overwrite them with
%    the appropriate value.  But we'll leave the cells with
%    incomplete data support alone (i.e. the ultimate leading and
%    lagging cells).  That way, targets there won't be able to
%    cross the thresholds.  It may be desirable to use partial data
%    support with modified thresholds.
thresholdMap = inf * ones(nDopBin, nRngBin);

indxFirstFullSupport = nLead + nGuardLead + 1;
indxLastFullSupport = nRngBin - nLag - nGuardLag;

fprintf('Performing CFAR...\n');

% Create the threshold range-Doppler map
% Loop over doppler bins
for iDop = 1 : 1 : nDopBin
    if(mod(iDop,100) == 0)
        fprintf('\tDoppler bin #%i (of #%i)\n', iDop, nDopBin);
    end
    % Loop over range bins (to test)
    for iRng = indxFirstFullSupport : 1 : indxLastFullSupport
        idxLead = iRng - nGuardLead - nLead : 1 : iRng - nGuardLead - 1;
        idxLag = iRng + nGuardLag + 1 : 1 : iRng + nGuardLag + nLag;
        bins = abs(rdMap(iDop, [idxLead idxLag]));

        thresholdMap(iDop, iRng) = computeThreshold(alphaScale, bins);
    end % End loop iRng
end % End loop iDop
else
    C = abs([zeros(nDopBin,nLead+nGuardLead) rdMap zeros(nDopBin,nLag+nGuardLag)]);
    filt = [ones(1,nLead) zeros(1,nGuardLead+nGuardLag+1) ones(1,nLag)];
    D = filter(filt,sum(filt),C,[],2);
    thresholdMap = D(:,(nLead+nGuardLead+nLag+nGuardLag+1):end)*alphaScale;
    thresholdMap(:,[1:(nLead+nGuardLead), (end-nLag-nGuardLag+1):end]) = inf;
end

% Find threshold crossings
[dop, rng] = find(abs(rdMap) > abs(thresholdMap));
if size(rdMap,1) == 1
    dop = dop.';
    rng = rng.';
end
detections = [rng dop];

% Visualize results
if(debug>0)
    figure; imagesc(20*log10(abs(thresholdMap)));
    xlabel('Range Bin');
    ylabel('Doppler Bin');
    title('Threshold Map')

    figure; imagesc(20*log10(abs(rdMap)));
    xlabel('Range Bin');
    ylabel('Doppler Bin');
    title('Range-Doppler Map')
    hold on;
        plot(detections(:,1), detections(:,2), 'ro', 'MarkerSize', 8);
    end
if nargout < 2
    clear thresholdMap
end
end % End function cacfar1d

% Begin helper function computeAlphaScale
%
% Compute the scale parameter, alpha, above the underlying
% background necessary to declare a detection
function alphaScale = computeAlphaScale(pfa, nBin)
    alphaScale = nBin * (pfa^(-1/nBin) - 1);
end % End function computeAlphaScale

% Begin helper function computeThreshold
% Compute the threshold necessary to declare a detection based on
% the average value of the surrounding interference and increased
% by the scale parameter, alpha
function threshold = computeThreshold(alphaScale, bins)
    nBin = length(bins);
    threshold = (alphaScale / nBin) * sum(bins);
end % End function computeThreshold
