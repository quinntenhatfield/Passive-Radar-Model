function clusters = combineExceedances (raw,rThresh,dThresh)
if isempty(raw)
    clusters = {};
    return
end

%%
%generate "strings" of exceedances along "range"direction
dops = unique(raw(:,2));
strings = {};

for ii = 1:numel(dops)
    rngs = sort(raw(raw(:,2)==dops(ii)),1);
    rDif = diff(rngs);
    idx = find(rDif>rThresh);
    if isempty(idx)
        idx = [0,numel(rngs)];
    else
        idx = [0,idx.',numel(rngs)];
    end
    for jj = 1:numel(idx)-1
        strings{end+1} = rngs((idx(jj)+1):idx(jj+1));
        strings{end}(:,2) = dops(ii);
    end
end
    
% to be honest, I'm not sure that last step is really worth doing, but I'll
% get over it.

% now group strings into clusters
clusters = strings(1); % initialize
for ii = 2:numel(strings)
    matchFound = false;
    for jj = 1:numel(clusters)
        if any(abs(bsxfun(@plus,clusters{jj}(:,1),-strings{ii}(:,1)'))<rThresh & ...
               abs(bsxfun(@plus,clusters{jj}(:,2),-strings{ii}(:,2)'))<dThresh)
           clusters{jj} = [clusters{jj};strings{ii}];
           matchFound = true;
           break
        end
    end
    if ~matchFound
        clusters{end+1} = strings{ii};
    end
end

if false
%% debug plot
fh = figure,
ah = axes;
hold all, grid on,
ph_raw = plot(raw(:,1),raw(:,2),'k.','DisplayName','EXC');
ph_str = [];
for ii = 1:numel(strings)
    ph_str(ii) = plot(strings{ii}(:,1),strings{ii}(:,2),'o','DisplayName','STR');
end
ph_clu = [];
for ii = 1:numel(clusters)
    ph_clu(ii) = plot(clusters{ii}(:,1),clusters{ii}(:,2),'x','DisplayName','CLU');
end
lh = legend([ph_raw,ph_str(1),ph_clu]);
set([ph_clu ph_str],'MarkerSize',14,'LineWidth',3);

%%
end
end