% Get stim starts and stops
stimTimes = table();
dataFiles1 = fileSearch('H:\LSD+stim_persist\mTOR\processed','.mat');
c = 1;
for ii = 1:numel(dataFiles1)
    if contains(dataFiles1{ii},'sIL')
        load(['H:\LSD+stim_persist\mTOR\processed\',dataFiles1{ii}], ...
            'hist')
        stimTimes.ID{c} = dataFiles1{ii};
        % Find first value over 600
        firstValueIndex = find(hist.eventTs.t{9} >= 600, 1, 'first');
        stimTimes.start(c) = hist.eventTs.t{9}(firstValueIndex);
        stimTimes.stop(c) = hist.eventTs.t{10}(end);
        c = c+1;
    end  
end
%
dataFiles2 = fileSearch('H:\LSD+stim_persist\1HrStim','.mat');
for ii = 1:numel(dataFiles2)
    if contains(dataFiles2{ii},'sIL')
        load(['H:\LSD+stim_persist\1HrStim\',dataFiles2{ii}], ...
            'hist')
        stimTimes.ID{c} = dataFiles2{ii};
        % Find first value over 600
        firstValueIndex = find(hist.eventTs.t{9} >= 600, 1, 'first');
        stimTimes.start(c) = hist.eventTs.t{9}(firstValueIndex);
        stimTimes.stop(c) = hist.eventTs.t{10}(end);
        c = c+1;
    end  
end
%
dataFiles3 = fileSearch('H:\LSD+stim_persist\2HrStim','.mat');
for ii = 1:numel(dataFiles3)
    if contains(dataFiles3{ii},'sIL')
        load(['H:\LSD+stim_persist\2HrStim\',dataFiles3{ii}], ...
            'hist')
        stimTimes.ID{c} = dataFiles3{ii};
        if numel(hist.eventTs.t)<9
            stimTimes.start(c) = hist.eventTs.t{4};
            stimTimes.stop(c) = hist.eventTs.t{5};
        else
            % Find first value over 600
            firstValueIndex = find(hist.eventTs.t{9} >= 600, 1, 'first');
            stimTimes.start(c) = hist.eventTs.t{9}(firstValueIndex);
            stimTimes.stop(c) = hist.eventTs.t{10}(end);
        end
        c = c+1;
    end  
end
save('H:\LSD+stim_persist\publicationData\movementData.mat','stimTimes')
%% Grab movement data
files = fileSearch('H:\LSD+stim_persist\1Hr_preVpost\data\movement','.csv')';
% Convert secs to frames
time = 60*30;
% Pre-allocate
[dist,tot,vel] = cell(1,numel(files));
for ii = 1:numel(files)
    this = readtable(files{ii});
    % NaN data with likelihoods below .8
    for jj = 4:3:28
        inds = table2array(this(:,jj)) < 0.8;
        this(inds,jj-2) = table(NaN);
        this(inds,jj-1) = table(NaN);
    end
    % Get centroid defined by tail, hip, spine, shoulder
    centX = mean(table2array(this(:,2:3:11)),2,'omitnan');
    centY = mean(table2array(this(:,3:3:12)),2,'omitnan');
    for jj = 1:numel(centX)-1
        dist{ii}(jj) = sqrt((centX(jj+1)-centX(jj))^2+...
            ((centY(jj+1)-centY(jj))^2));
    end
    % Calculate total movement over time (sec)
    ints = floor(numel(dist{ii})/time);
    for jj = 1:ints
        tot{ii}(jj) = sum(dist{ii}(time*jj-time+1:time*jj),'omitnan');
    end
    % Calculate velocity (frames/sec)
    vel{ii} = tot{ii}./time;
end
save('H:\LSD+stim_persist\publicationData\movementData.mat','-append','vel','files')
%% Prepare movement data
load('H:\LSD+stim_persist\publicationData\movementData.mat')
lsdInds = find(contains(files,'LSD-sIL'));
salInds = find(contains(files,'SAL-sIL'));
baseInds = find(contains(files,'base_') | contains(files,'baseSAL'));

lsdPre = NaN(numel(lsdInds),10);
lsdPost = NaN(numel(lsdInds),70);
for ii = 1:numel(lsdInds)
    thisInd = find(strcmp(stimTimes.ID,[files{lsdInds(ii)}(1:end-4), ...
        '_all.mat']));
    stop = floor(stimTimes.stop(thisInd)/60);
    lsdPre(ii,:) = vel{lsdInds(ii)}(1:10);
    len = numel(vel{lsdInds(ii)}(stop:end));
    lsdPost(ii,1:len) = vel{lsdInds(ii)}(stop:end);
end

lsd24Inds = find(contains(files,'24PostLSD'));
lsd24Post = NaN(numel(lsd24Inds),30);
for ii = 1:numel(lsd24Inds)
    lsd24Post(ii,:) = vel{lsd24Inds(ii)}(1:30);
end

lsd48Inds = find(contains(files,'48PostLSD'));
lsd48Post = NaN(numel(lsd48Inds),30);
for ii = 1:numel(lsd48Inds)
    lsd48Post(ii,:) = vel{lsd48Inds(ii)}(1:30);
end

lsd72Inds = find(contains(files,'72PostLSD'));
lsd72Post = NaN(numel(lsd72Inds),30);
for ii = 1:numel(lsd72Inds)
    lsd72Post(ii,:) = vel{lsd72Inds(ii)}(1:30);
end

lsd144Inds = find(contains(files,'144PostLSD'));
lsd144Post = NaN(numel(lsd144Inds),30);
for ii = 1:numel(lsd144Inds)
    lsd144Post(ii,:) = vel{lsd144Inds(ii)}(1:30);
end

lsd168Inds = find(contains(files,'168PostLSD'));
lsd168Post = NaN(numel(lsd168Inds),30);
for ii = 1:numel(lsd168Inds)
    lsd168Post(ii,:) = vel{lsd168Inds(ii)}(1:30);
end

salPre = NaN(numel(salInds),10);
salPost = NaN(numel(salInds),70);
for ii = 1:numel(salInds)
    thisInd = find(strcmp(stimTimes.ID,[files{salInds(ii)}(1:end-4), ...
        '_all.mat']));
    stop = floor(stimTimes.stop(thisInd)/60);
    salPre(ii,:) = vel{salInds(ii)}(1:10);
    len = numel(vel{salInds(ii)}(stop:end));
    salPost(ii,1:len) = vel{salInds(ii)}(stop:end);
end

sal24Inds = find(contains(files,'24PostSAL'));
sal24Post = NaN(numel(sal24Inds),30);
for ii = 1:numel(sal24Inds)
    sal24Post(ii,:) = vel{sal24Inds(ii)}(1:30);
end

sal48Inds = find(contains(files,'48PostSAL'));
sal48Post = NaN(numel(sal48Inds),30);
for ii = 1:numel(sal48Inds)
    sal48Post(ii,:) = vel{sal48Inds(ii)}(1:30);
end

sal72Inds = find(contains(files,'72PostSAL'));
sal72Post = NaN(numel(sal72Inds),30);
for ii = 1:numel(sal72Inds)
    sal72Post(ii,:) = vel{sal72Inds(ii)}(1:30);
end

sal144Inds = find(contains(files,'144PostSAL'));
sal144Post = NaN(numel(sal144Inds),30);
for ii = 1:numel(sal144Inds)
    sal144Post(ii,:) = vel{sal144Inds(ii)}(1:30);
end

sal168Inds = find(contains(files,'168PostSAL'));
sal168Post = NaN(numel(sal168Inds),30);
for ii = 1:numel(sal168Inds)
    sal168Post(ii,:) = vel{sal168Inds(ii)}(1:30);
end

basePre = NaN(numel(baseInds),15);
basePost = NaN(numel(baseInds),15);
for ii = 1:numel(baseInds)
    basePre(ii,:) = vel{baseInds(ii)}(1:15);
    basePost(ii,:) = vel{salInds(ii)}(16:30);
end

base24Inds = find(contains(files,'base24'));
base24Post = NaN(numel(base24Inds),30);
for ii = 1:numel(base24Inds)
    base24Post(ii,:) = vel{base24Inds(ii)}(1:30);
end

base48Inds = find(contains(files,'base48'));
base48Post = NaN(numel(base48Inds),30);
for ii = 1:numel(base48Inds)
    base48Post(ii,:) = vel{base48Inds(ii)}(1:30);
end

base96Inds = find(contains(files,'base96'));
base96Post = NaN(numel(base96Inds),30);
for ii = 1:numel(base96Inds)
    base96Post(ii,:) = vel{base96Inds(ii)}(1:30);
end
%% Build pre vs post models; test on washout days
% Set permutations
nPerms = 100;
% BASE
[aBase,aBasePerm,aBase24,aBase48,aBase96] = deal(NaN(1,nPerms));
for jj = 1:nPerms
    trainPost = [];
    trainPre = [];
    testPost = [];
    testPre = [];
    trainY = [];
    testY = [];
    for ii = 1:height(basePre)
        this = basePost(ii,:);
        this(isnan(this)) = [];
        if any(this)
            trainPostInds = randperm(numel(this),floor(numel(this)*0.8));
            testPostInds = ~ismember(1:numel(this),trainPostInds);
            trainPost = [trainPost;basePost(ii,trainPostInds)']; %#ok<*AGROW>
            testPost = [testPost;basePost(ii,testPostInds)'];

            trainPreInds = randperm(10,8);
            testPreInds = ~ismember(1:10,trainPreInds);
            trainPre = [trainPre;basePre(ii,trainPreInds)'];
            testPre = [testPre;basePre(ii,testPreInds)'];

            trainY = [trainY;zeros(numel(trainPreInds),1);
                ones(numel(trainPostInds),1)];
            testY = [testY;zeros(sum(testPreInds),1);
                ones(sum(testPostInds),1)];
        end
    end
    mdl = fitglm([trainPre;trainPost],trainY,'distribution','binomial');
    scores = predict(mdl,[testPre;testPost]);
    [~,~,~,aBase(jj)] = perfcurve(testY,scores,1);
    % Permuted
    mdlPerm = fitglm([trainPre;trainPost],trainY(randperm(numel(trainY),...
        numel(trainY))),'distribution','binomial');
    scores = predict(mdlPerm,[testPre;testPost]);
    [~,~,~,aBasePerm(jj)] = perfcurve(testY,scores,1);

    % Apply model to 24 hours post
    testPost24 = base24Post(randperm(numel(base24Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost24']);
    [~,~,~,aBase24(jj)] = perfcurve(testY,scores,1);

    % Apply model to 48 hours post
    testPost48 = base48Post(randperm(numel(base48Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost48']);
    [~,~,~,aBase48(jj)] = perfcurve(testY,scores,1);

    % Apply model to 96 hours post
    testPost96 = base96Post(randperm(numel(base96Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost96']);
    [~,~,~,aBase96(jj)] = perfcurve(testY,scores,1);
end
% LSD
[aLSD,aLSDPerm,aLSD24,aLSD48,aLSD72,aLSD144,aLSD168] = deal(NaN(1,nPerms));
for jj = 1:nPerms
    trainPost = [];
    trainPre = [];
    testPost = [];
    testPre = [];
    trainY = [];
    testY = [];
    for ii = 1:height(lsdPre)
        this = lsdPost(ii,:);
        this(isnan(this)) = [];
        if any(this)
            trainPostInds = randperm(numel(this),floor(numel(this)*0.8));
            testPostInds = ~ismember(1:numel(this),trainPostInds);
            trainPost = [trainPost;lsdPost(ii,trainPostInds)'];
            testPost = [testPost;lsdPost(ii,testPostInds)'];

            trainPreInds = randperm(10,8);
            testPreInds = ~ismember(1:10,trainPreInds);
            trainPre = [trainPre;lsdPre(ii,trainPreInds)'];
            testPre = [testPre;lsdPre(ii,testPreInds)'];

            trainY = [trainY;zeros(numel(trainPreInds),1);
                ones(numel(trainPostInds),1)];
            testY = [testY;zeros(sum(testPreInds),1);
                ones(sum(testPostInds),1)];
        end
    end
    mdl = fitglm([trainPre;trainPost],trainY,'distribution','binomial');
    scores = predict(mdl,[testPre;testPost]);
    [~,~,~,aLSD(jj)] = perfcurve(testY,scores,1);

    % Permuted
    mdlPerm = fitglm([trainPre;trainPost],trainY(randperm(numel(trainY),...
        numel(trainY))),'distribution','binomial');
    scores = predict(mdlPerm,[testPre;testPost]);
    [~,~,~,aLSDPerm(jj)] = perfcurve(testY,scores,1);

    % Apply model to 24 hours post
    testPost24 = lsd24Post(randperm(numel(lsd24Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost24']);
    [~,~,~,aLSD24(jj)] = perfcurve(testY,scores,1);

    % Apply model to 48 hours post
    testPost48 = lsd48Post(randperm(numel(lsd48Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost48']);
    [~,~,~,aLSD48(jj)] = perfcurve(testY,scores,1);

    % Apply model to 72 hours post
    testPost72 = lsd72Post(randperm(numel(lsd72Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost72']);
    [~,~,~,aLSD72(jj)] = perfcurve(testY,scores,1);

    % Apply model to 144 hours post
    testPost144 = lsd144Post(randperm(numel(lsd144Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost144']);
    [~,~,~,aLSD144(jj)] = perfcurve(testY,scores,1);

    % Apply model to 168 hours post
    testPost168 = lsd168Post(randperm(numel(lsd168Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost168']);
    [~,~,~,aLSD168(jj)] = perfcurve(testY,scores,1);
end

% SAL
[aSAL,aSALPerm,aSAL24,aSAL48,aSAL72,aSAL144,aSAL168] = deal(NaN(1,nPerms));
for jj = 1:nPerms
    trainPost = [];
    trainPre = [];
    testPost = [];
    testPre = [];
    trainY = [];
    testY = [];
    for ii = 1:height(salPre)
        this = salPost(ii,:);
        this(isnan(this)) = [];
        if any(this)
            trainPostInds = randperm(numel(this),floor(numel(this)*0.8));
            testPostInds = ~ismember(1:numel(this),trainPostInds);
            trainPost = [trainPost;salPost(ii,trainPostInds)'];
            testPost = [testPost;salPost(ii,testPostInds)'];

            trainPreInds = randperm(10,8);
            testPreInds = ~ismember(1:10,trainPreInds);
            trainPre = [trainPre;salPre(ii,trainPreInds)'];
            testPre = [testPre;salPre(ii,testPreInds)'];

            trainY = [trainY;zeros(numel(trainPreInds),1);ones(numel(trainPostInds),1)];
            testY = [testY;zeros(sum(testPreInds),1);ones(sum(testPostInds),1)];
        end
    end
    mdl = fitglm([trainPre;trainPost],trainY,'distribution','binomial');
    scores = predict(mdl,[testPre;testPost]);
    [~,~,~,aSAL(jj)] = perfcurve(testY,scores,1);

    % Permuted
    mdlPerm = fitglm([trainPre;trainPost],trainY(randperm(numel(trainY),...
        numel(trainY))),'distribution','binomial');
    scores = predict(mdlPerm,[testPre;testPost]);
    [~,~,~,aSALPerm(jj)] = perfcurve(testY,scores,1);

    % Apply model to 24 hours post
    testPost24 = sal24Post(randperm(numel(sal24Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost24']);
    [~,~,~,aSAL24(jj)] = perfcurve(testY,scores,1);

    % Apply model to 48 hours post
    testPost48 = sal48Post(randperm(numel(sal48Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost48']);
    [~,~,~,aSAL48(jj)] = perfcurve(testY,scores,1);

    % Apply model to 72 hours post
    testPost72 = sal72Post(randperm(numel(sal72Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost72']);
    [~,~,~,aSAL72(jj)] = perfcurve(testY,scores,1);

    % Apply model to 144 hours post
    testPost144 = sal144Post(randperm(numel(sal144Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost144']);
    [~,~,~,aSAL144(jj)] = perfcurve(testY,scores,1);

    % Apply model to 168 hours post
    testPost168 = sal168Post(randperm(numel(sal168Post),numel(testPost)));
    scores = predict(mdl,[testPre;testPost168']);
    [~,~,~,aSAL168(jj)] = perfcurve(testY,scores,1);
end
save('H:\LSD+stim_persist\publicationData\movementData.mat','-append', ...
    'aBase','aBasePerm','aBase24','aBase48','aBase96','aLSD','aLSDPerm', ...
    'aLSD24','aLSD48','aLSD72','aLSD144','aLSD168','aSAL','aSALPerm', ...
    'aSAL24','aSAL48','aSAL72','aSAL144','aSAL168')