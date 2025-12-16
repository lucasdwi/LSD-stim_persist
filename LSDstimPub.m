%% Sham stimulation models
% Load data
[data2,samps2,files2] = collateData('H:\LSD+stim_persist\processed\SAL\',...
    {'base'},{'pow','coh'},'trl','raw');
[data3,samps3,files3] = collateData('H:\LSD+stim_persist\2HrStim\',...
    {'base'},{'pow','coh'},'trl','raw');

% Combine data
data = cat(1,data2{1},data3{1});
samps = cat(1,samps2{1},samps3{1});
files = cat(1,files2{1},files3{1});
allData = [];
for ii = 1:numel(files)
    parts = strsplit(files{ii},'_');
    nameParts = strsplit(parts{1},'-');
    % Full Id
    ratID = parts(1);
    % ID
    baseID = nameParts{1};
    % Group
    group = nameParts{2};
    % Get date
    thisDate = datetime(parts{3});
    % Get data
    thisData = data{ii};
    thisSamp = samps{ii};
    % Set pre to be first 10 minutes
    preInds = thisSamp<=600;
    % Set post inds to be after pre
    postInds = thisSamp>600;

    % Pre
    preData = thisData(preInds,:);
    preSamps = thisSamp(preInds);
    allData = [allData;table(preSamps',...
        repmat(ratID,size(preData,1),1),...
        repmat({baseID},size(preData,1),1),...
        repmat({group},size(preData,1),1),...
        repmat(thisDate,size(preData,1),1),...
        repmat({'pre'},size(preData,1),1),preData,...
        'VariableNames',{'time','fullID','ID','group',...
        'date','type','data'})]; %#ok<*AGROW>
    % Post
    postData = thisData(postInds,:);
    postSamps = thisSamp(postInds);
    allData = [allData;table(postSamps',...
        repmat(ratID,size(postData,1),1),...
        repmat({baseID},size(postData,1),1),...
        repmat({group},size(postData,1),1),...
        repmat(thisDate,size(postData,1),1),...
        repmat({'post'},size(postData,1),1),postData,...
        'VariableNames',{'time','fullID','ID','group',...
        'date','type','data'})]; 
end

% Normalize power
% Get starts and stops for normalizing power
normInds = [1:6:48; 6:6:48]';
% Set up empty table for all data
allData.normData = allData.data;
for ii = 1:size(allData,1)
    if mod(ii,1000) == 0
        disp(ii)
    end
    % Set up empty thisNorm vector
    thisNorm = zeros(1,48);
    for k = 1:size(normInds,1)
        thisSum = sum(allData.data(ii,normInds(k,1):normInds(k,2)));
        thisNorm(normInds(k,1):normInds(k,2)) = allData.data(ii,...
            normInds(k,1):normInds(k,2))./thisSum;
    end
    allData.normData(ii,1:48) = thisNorm;
end

% Separate data
% Set washout times
times2 = {'24','48','96'};
% Set up simple ID vectors in the same order as the data
uID = unique(allData.fullID);
uShortID = unique(cellfun(@(x) x{1},cellfun(@(x) strsplit(x,'-'),uID,...
    'uniformoutput',false),'uniformoutput',false));
ids = uShortID';
% Preallocate
[pre,post] = deal(cell(numel(ids,1)));
wash = cell(numel(ids,numel(times2)));
for jj = 1:numel(ids)
    % Get Pre data
    pre{jj} = allData(contains(allData.fullID,ids{jj}) & ...
        strcmp(allData.type,'pre'),:);
    % Get Post data
    post{jj} = allData(contains(allData.fullID,ids{jj}) & ...
        strcmp(allData.type,'post'),:);
    % Get Wash data
    for ii = 1:numel(times2)
        wash{jj,ii} = allData(contains(allData.fullID,ids{jj}) & ...
            contains(allData.group,times2{ii}),:);
    end
end
% Remove rats with no washout data
pre(8) = [];
post(8) = [];
wash(8,:) = [];
ids(8) = [];

% Save 
save('H:\LSD+stim_persist\publicationData\shamStimData.mat','pre','post', ...
    'wash','ids')
%% Build and test sham-stimulation models
% Load data
load('H:\LSD+stim_persist\publicationData\shamStimData.mat')

% Set washout times
times2 = {'24','48','96'};
% Set number of permutations
nPerms = 100;
% Preallocate
[mdl,mdlPerm] = deal(cell(1,nPerms));
[acc,baseA_glm80,baseA_glm80_perm] = deal(NaN(1,nPerms));
[baseA_glm80_wash,baseA_glm80_wash_perm] = deal(NaN(numel(times2),nPerms));
for k = 1:nPerms
    disp(k)
    %% Use up to 100 of all samples
    [subPre,subPost,postInd] = deal(cell(numel(ids),1));
    subWash = cell(numel(ids),numel(times2));
    for jj = 1:numel(ids)
        nPre2  = min([height(pre{jj}) 100]);
        nPost2 = min([height(post{jj}) 100]);
        subPre{jj} = pre{jj}.normData(randperm(height(pre{jj}), ...
            nPre2),:);

        postInd{jj} = randperm(height(post{jj}),nPost2);
        subPost{jj} = post{jj}.normData(postInd{jj},:);

        for ii = 1:numel(times2)
            nWash = min([height(wash{jj,ii}) 100]);
            subWash{jj,ii} = wash{jj,ii}.normData(randperm( ...
                height(wash{jj,ii}),nWash),:);
        end
    end
    % split into 80:20 chunks
    [subPreTrain,subPostTrain,subPreTest,subPostTest] = ...
        deal(cell(numel(ids),1));
    for jj = 1:numel(ids)
        preTrainInd = randperm(height(subPre{jj}), ...
            round(height(subPre{jj}).*.8));
        preTestInd = find(~ismember(1:height(subPre{jj}), ...
            preTrainInd));
        postTrainInd = randperm(height(subPost{jj}), ...
            round(height(subPost{jj}).*.8));
        postTestInd = find(~ismember(1:height(subPost{jj}), ...
            postTrainInd));

        subPreTrain{jj} = subPre{jj}(preTrainInd,:);
        subPostTrain{jj} = subPost{jj}(postTrainInd,:);

        % Make sure test data has equal contributions from each rat
        subPreTest{jj} = subPre{jj}(preTestInd(randperm(numel( ...
            preTestInd),...
            min([numel(preTestInd) numel(postTestInd)]))),:);
        subPostTest{jj} = subPost{jj}(postTestInd(randperm( ...
            numel(postTestInd),...
            min([numel(preTestInd) numel(postTestInd)]))),:);
    end
    % calculate weights
    nPre2 = cellfun(@height,subPreTrain);
    nPost2 = cellfun(@height,subPostTrain);
    preW = cell(numel(ids),1);
    postW = cell(numel(ids),1);
    for jj = 1:numel(ids)
        preW{jj} = repmat((0.5/numel(ids))./nPre2(jj), ...
            nPre2(jj),1);
        postW{jj} = repmat((0.5/numel(ids))./nPost2(jj), ...
            nPost2(jj),1);
    end
    % pre vs. post
    trainX = cat(1,subPostTrain{:},subPreTrain{:});
    trainY = [ones(height(cat(1,subPostTrain{:})),1); ...
        zeros(height(cat(1,subPreTrain{:})),1)];
    testX = cat(1,subPostTest{:},subPreTest{:});
    testY = [ones(height(cat(1,subPostTest{:})),1); ...
        zeros(height(cat(1,subPreTest{:})),1)];
    weight = cat(1,postW{:},preW{:});
    mdl{k} = fitglm(trainX,trainY,'distribution','binomial', ...
        'weights',weight);
    % permuted model
    permInd = randperm(numel(trainY),numel(trainY));
    permTrainY = trainY(permInd);
    permWeight = weight(permInd);
    mdlPerm{k} = fitglm(trainX,permTrainY,'distribution',...
        'binomial','weights',permWeight);
    % Test
    scores = predict(mdl{k},testX);
    [~,~,~,baseA_glm80(k)] = perfcurve(testY,scores,1);
    % Accuracies
    this = round(scores);
    acc(k) = mean(this==testY);
    % Permuted
    scores = predict(mdlPerm{k},testX);
    [~,~,~,baseA_glm80_perm(k)] = perfcurve(testY,scores,1);
    

    % applied to wash
    % Grab an equal number of wash data as preTest data
    [washTestX,washTestY] = deal(cell(numel(ids),numel(times2)));
    for ii = 1:numel(times2)
        for jj = 1:numel(ids)
            if ~isempty(wash{jj,ii})
                washTestX{jj,ii} = [subPreTest{jj};wash{jj,ii}.normData( ...
                    randperm(...
                    height(wash{jj,ii}), min([height(wash{jj,ii}) ...
                    height(subPostTest{jj})])),:)];
                washTestY{jj,ii} = [zeros(height(subPreTest{jj}),1); ...
                    ones(height(subPreTest{jj}),1)];
            end
        end
        % Get real AUCs
        scores = predict(mdl{k},cat(1,washTestX{:,ii}));
        [~,~,~,baseA_glm80_wash(ii,k)] = perfcurve(cat(1,washTestY{:,ii}), ...
            scores,1);
        % Get permuted AUCs
        scores = predict(mdlPerm{k},cat(1,washTestX{:,ii}));
        [~,~,~,baseA_glm80_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestY{:,ii}),scores,1);
    end
end
% Save
save('H:\LSD+stim_persist\publicationData\shamStimMdl.mat','baseA_glm80', ...
    'baseA_glm80_perm','baseA_glm80_wash','baseA_glm80_wash_perm')
%% Prepare cohort 1 data - 1 hr stim
% Load files
[data1,samps1,files1] = collateData('H:\LSD+stim_persist\1HrStim\', ...
    {'-sIL'},{'pow','coh'},'trl','raw');
[data2,samps2,files2] = collateData('H:\LSD+stim_persist\1HrStim\', ...
    {'Post'},{'pow','coh'},'trl','raw');
[data3,samps3,files3] = collateData('H:\LSD+stim_persist\mTOR\processed\', ...
    {'-sIL'},{'pow','coh'},'trl','raw');
% Combine
data = cat(1,data1{1},data2{1},data3{1}); ...
    samps = cat(1,samps1{1},samps2{1},samps3{1}); ...
    files = cat(1,files1{1},files2{1},files3{1});

% Organize data
info = cell(numel(files,1),2);
for ii = 1:numel(files)
    parts = strsplit(files{ii},'_');
    info{ii,1} = parts{1};
    info{ii,2} = parts{3};
end
uIDs = unique(info(:,1));
% Go through each recording
allData = [];
for ii = 1:numel(uIDs)
    thisRat = find(contains(info(:,1),uIDs{ii}));
    ratID = uIDs(ii);
    % Get date
    thisDate = datetime(info{thisRat,2});
    % Get ID and group info
    parts = strsplit(uIDs{ii},'-');
    baseID = parts{1};
    group = strjoin(parts(2:end),'-');
    % Get data
    thisData = data{thisRat};
    thisSamp = samps{thisRat};
    % Check if stim or post
    if contains(uIDs{ii},'sIL')
        % Load event data for stim time stamps
        if any(contains(files3{1},uIDs{ii}))
            load(['H:\LSD+stim_persist\mTOR\processed\',files{thisRat}], ...
                'hist')
        else
            load(['H:\LSD+stim_persist\1HrStim\',files{thisRat}],'hist')
        end
        % Set stim event index
        eInd = 9; 
        % Get stim start and stops
        stimBegin = hist.eventTs.t{eInd}(1);
        stimEnd = hist.eventTs.t{eInd}(end);
        % Find all breaks; add in last one and slight jitter (0.05) to get
        % beyond stim
        stimInds = logicFind(1,round(diff(hist.eventTs.t{eInd}))>0,'==');
        stimIntStart = [hist.eventTs.t{eInd}(stimInds);stimEnd]+0.05;
        % Use the next instance of stim (ind+1) to get ends; add in last
        % end
        stimIntStop = [hist.eventTs.t{eInd}(stimInds+1);...
            stimIntStart(end)+10];
        % Check for stim at start of recording, if exists, remove from stim
        % data and use to set pre times
        if stimIntStart(1)<600
            preStart = stimIntStart(1);
            preStop = stimIntStop(1);
            % Remove from stimInt
            stimIntStart(1) = [];
            stimIntStop(1) = [];
            % Reset stimBegin
            stimBegin = preStop;
        else
            preStart = 0;
            preStop = stimBegin;
        end
        postStart = stimIntStop(end);
        postStop = samps{thisRat}(end)+1.5;

        % Split data into preStim, stim, and postStim
        % Pre
        preInds = thisSamp>preStart & thisSamp<preStop;
        preData = thisData(preInds,:);
        preSamps = thisSamp(preInds);
        stimTime = preSamps-stimBegin;
        allData = [allData;table(preSamps',stimTime',...
            repmat(ratID,size(preData,1),1),...
            repmat({baseID},size(preData,1),1),...
            repmat({group},size(preData,1),1),...
            repmat(thisDate,size(preData,1),1),...
            repmat({'pre'},size(preData,1),1),preData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
        % Stim
        this = NaN(numel(stimIntStart),numel(thisSamp));
        for jj = 1:numel(stimIntStart)
            this(jj,:) = thisSamp>stimIntStart(jj) & ...
                thisSamp<stimIntStop(jj);
        end
        stimInds = any(this,1);
        stimData = thisData(stimInds,:);
        stimSamps = thisSamp(stimInds);
        stimTime = stimSamps-stimBegin;
        allData = [allData;table(stimSamps',stimTime',...
            repmat(ratID,size(stimData,1),1),...
            repmat({baseID},size(stimData,1),1),...
            repmat({group},size(stimData,1),1),...
            repmat(thisDate,size(stimData,1),1),...
            repmat({'stim'},size(stimData,1),1),stimData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
        % Post
        postInds = thisSamp>postStart & thisSamp<postStop;
        postData = thisData(postInds,:);
        postSamps = thisSamp(postInds);
        stimTime = postSamps-stimEnd;
        allData = [allData;table(postSamps',stimTime',...
            repmat(ratID,size(postData,1),1),...
            repmat({baseID},size(postData,1),1),...
            repmat({group},size(postData,1),1),...
            repmat(thisDate,size(postData,1),1),...
            repmat({'post'},size(postData,1),1),postData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
    elseif contains(uIDs{ii},'Post')
        % Get wash data
        thisData = data{thisRat};
        thisSamp = samps{thisRat};
        thisDate = datetime(info{thisRat,2});
        % Get number of hours post stim from file name
        parts = strsplit(uIDs{ii},'-');
        % Convert hours to seconds
        secs = sscanf(parts{2},'%f')*3600;
        % Add to time
        stimTime = thisSamp+secs;
        allData = [allData;table(thisSamp',stimTime',...
            repmat(ratID,size(thisData,1),1),...
            repmat({baseID},size(thisData,1),1),...
            repmat({group},size(thisData,1),1),...
            repmat(thisDate,size(thisData,1),1),...
            repmat({'wash'},size(thisData,1),1),thisData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
    else
        % If sham, set sham to start at 10 minutes and end 1 hour later
        if contains(uIDs{ii},'sham')
            % Set pre to be first 10 minutes
            preInds = thisSamp<=600;
            % Set sham inds to be 1 hour after pre
            shamInds = thisSamp>600 & thisSamp<=4200;
            % Set post inds to be after sham
            postInds = thisSamp>4200;

            % Pre
            preData = thisData(preInds,:);
            preSamps = thisSamp(preInds);
            stimTime = preSamps-600;
            allData = [allData;table(preSamps',stimTime',...
                repmat(ratID,size(preData,1),1),...
                repmat({baseID},size(preData,1),1),...
                repmat({group},size(preData,1),1),...
                repmat(thisDate,size(preData,1),1),...
                repmat({'pre'},size(preData,1),1),preData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})]; 
            % Sham
            shamData = thisData(shamInds,:);
            shamSamps = thisSamp(shamInds);
            shamTime = shamSamps-600;
            allData = [allData;table(shamSamps',shamTime',...
                repmat(ratID,size(shamData,1),1),...
                repmat({baseID},size(shamData,1),1),...
                repmat({group},size(shamData,1),1),...
                repmat(thisDate,size(shamData,1),1),...
                repmat({'sham'},size(shamData,1),1),shamData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})]; 
            % Post
            postData = thisData(postInds,:);
            postSamps = thisSamp(postInds);
            stimTime = postSamps-4200;
            allData = [allData;table(postSamps',stimTime',...
                repmat(ratID,size(postData,1),1),...
                repmat({baseID},size(postData,1),1),...
                repmat({group},size(postData,1),1),...
                repmat(thisDate,size(postData,1),1),...
                repmat({'post'},size(postData,1),1),postData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})]; 
        else
            % Get baseline 'pre' data
            thisData = data{thisRat};
            thisSamp = samps{thisRat};
            thisDate = datetime(info{thisRat,2});
            % Add to time
            stimTime = NaN(height(thisData),1);
            allData = [allData;table(thisSamp',stimTime,...
                repmat(ratID,size(thisData,1),1),...
                repmat({baseID},size(thisData,1),1),...
                repmat({group},size(thisData,1),1),...
                repmat(thisDate,size(thisData,1),1),...
                repmat({'pre'},size(thisData,1),1),thisData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})]; 
        end
    end
end

% Normalize power
% Get starts and stops for normalizing power
normInds = [1:6:48; 6:6:48]';
% Set up empty table for all data
allData.normData = allData.data;
for ii = 1:size(allData,1)
    if mod(ii,1000) == 0
        disp(ii)
    end
    % Set up empty thisNorm vector
    thisNorm = zeros(1,48);
    for k = 1:size(normInds,1)
        thisSum = sum(allData.data(ii,normInds(k,1):normInds(k,2)));
        thisNorm(normInds(k,1):normInds(k,2)) = allData.data(ii,...
            normInds(k,1):normInds(k,2))./thisSum;
    end
    allData.normData(ii,1:48) = thisNorm;
end

% Set up simple ID vectors in the same order as the data
salData = allData(contains(allData.fullID, '-SAL-') & ...
    (strcmp(allData.type, 'pre') | (strcmp(allData.type, 'stim') & ...
    allData.stimTime <= 600)), :);
uSAL = unique(salData.fullID);
uShortSAL = cellfun(@(x) x{1},cellfun(@(x) strsplit(x,'-'),uSAL,...
    'uniformoutput',false),'uniformoutput',false);
lsdData = allData(contains(allData.fullID, '-LSD-') & ...
    (strcmp(allData.type, 'pre') | (strcmp(allData.type, 'stim') & ...
    allData.stimTime <= 600)), :);
uLSD = unique(lsdData.fullID);
uShortLSD = cellfun(@(x) x{1},cellfun(@(x) strsplit(x,'-'),uLSD,...
    'uniformoutput',false),'uniformoutput',false);
% Define rat IDs - 1 Hr stim
lsd_ids1 = uShortLSD';
sal_ids1 = uShortSAL';
% Save
save('H:\LSD+stim_persist\publicationData\cohort1-3Data.mat', ...
    'allData','data','files','samps','lsd_ids1','sal_ids1','uShortLSD', ...
    'uShortSAL')
%% Prepare cohort 2 data - 2 hr stim
% Load files
[data1,samps1,files1] = collateData('H:\LSD+stim_persist\2HrStim\',...
    {'-sIL'},{'pow','coh'},'trl','raw');
[data2,samps2,files2] = collateData('H:\LSD+stim_persist\2HrStim\',...
    {'Post'},{'pow','coh'},'trl','raw');
% Combine
data = cat(1,data1{1},data2{1});
samps = cat(1,samps1{1},samps2{1});
files = cat(1,files1{1},files2{1});

% Organize data
info = cell(numel(files,1),2);
for ii = 1:numel(files)
    parts = strsplit(files{ii},'_');
    info{ii,1} = parts{1};
    info{ii,2} = parts{3};
end
uIDs = unique(info(:,1));
% Go through each recording
allData = [];
for ii = 1:numel(uIDs)
    thisRat = find(contains(info(:,1),uIDs{ii}));
    ratID = uIDs(ii);
    % Get date
    thisDate = datetime(info{thisRat,2});
    % Get ID and group info
    parts = strsplit(uIDs{ii},'-');
    baseID = parts{1};
    group = strjoin(parts(2:end),'-');
    % Get data
    thisData = data{thisRat};
    thisSamp = samps{thisRat};
    % Check if stim or post
    if contains(uIDs{ii},'sIL')
        % Load stim events
        load(['H:\LSD+stim_persist\2HrStim\',files{thisRat}],'hist')
        % First set of files have different stim index
        if ii <=7
            eInd = 8;
        else
            eInd = 9;
        end
        % Get stim start and stops
        stimBegin = hist.eventTs.t{eInd}(1);
        stimEnd = hist.eventTs.t{eInd}(end);
        % Find all breaks; add in last one and slight jitter (0.05) to get
        % beyond stim
        stimInds = logicFind(1,round(diff(hist.eventTs.t{eInd}))>0,'==');
        stimIntStart = [hist.eventTs.t{eInd}(stimInds);stimEnd]+0.05;
        % Use the next instance of stim (ind+1) to get ends; add in last
        % end
        stimIntStop = [hist.eventTs.t{eInd}(stimInds+1);...
            stimIntStart(end)+10];
        % Check for stim at start of recording, if exists, remove from stim
        % data and use to set pre times
        if stimIntStart(1)<600
            preStart = stimIntStart(1);
            preStop = stimIntStop(1);
            % Remove from stimInt
            stimIntStart(1) = [];
            stimIntStop(1) = [];
            % Reset stimBegin
            stimBegin = preStop;
        else
            preStart = 0;
            preStop = stimBegin;
        end
        postStart = stimIntStop(end);
        postStop = samps{thisRat}(end)+1.5;
        % Split data into preStim, stim, and postStim
        % Pre
        preInds = thisSamp>preStart & thisSamp<preStop;
        preData = thisData(preInds,:);
        preSamps = thisSamp(preInds);
        stimTime = preSamps-stimBegin;
        allData = [allData;table(preSamps',stimTime',...
            repmat(ratID,size(preData,1),1),...
            repmat({baseID},size(preData,1),1),...
            repmat({group},size(preData,1),1),...
            repmat(thisDate,size(preData,1),1),...
            repmat({'pre'},size(preData,1),1),preData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})];
        % Stim
        this = NaN(numel(stimIntStart),numel(thisSamp));
        for jj = 1:numel(stimIntStart)
            this(jj,:) = thisSamp>stimIntStart(jj) & ...
                thisSamp<stimIntStop(jj);
        end
        stimInds = any(this,1);
        stimData = thisData(stimInds,:);
        stimSamps = thisSamp(stimInds);
        stimTime = stimSamps-stimBegin;
        allData = [allData;table(stimSamps',stimTime',...
            repmat(ratID,size(stimData,1),1),...
            repmat({baseID},size(stimData,1),1),...
            repmat({group},size(stimData,1),1),...
            repmat(thisDate,size(stimData,1),1),...
            repmat({'stim'},size(stimData,1),1),stimData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
        % Post
        postInds = thisSamp>postStart & thisSamp<postStop;
        postData = thisData(postInds,:);
        postSamps = thisSamp(postInds);
        stimTime = postSamps-stimEnd;
        allData = [allData;table(postSamps',stimTime',...
            repmat(ratID,size(postData,1),1),...
            repmat({baseID},size(postData,1),1),...
            repmat({group},size(postData,1),1),...
            repmat(thisDate,size(postData,1),1),...
            repmat({'post'},size(postData,1),1),postData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
    elseif contains(uIDs{ii},'Post')
        % Get wash data
        thisData = data{thisRat};
        thisSamp = samps{thisRat};
        thisDate = datetime(info{thisRat,2});
        % Get number of hours post stim from file name
        parts = strsplit(uIDs{ii},'-');
        % Convert hours to seconds
        secs = sscanf(parts{2},'%f')*3600;
        % Add to time
        stimTime = thisSamp+secs;
        allData = [allData;table(thisSamp',stimTime',...
            repmat(ratID,size(thisData,1),1),...
            repmat({baseID},size(thisData,1),1),...
            repmat({group},size(thisData,1),1),...
            repmat(thisDate,size(thisData,1),1),...
            repmat({'wash'},size(thisData,1),1),thisData,...
            'VariableNames',{'time','stimTime','fullID','ID','group',...
            'date','type','data'})]; 
    else
        % If sham, set sham to start at 10 minutes and end 1 hour later
        if contains(uIDs{ii},'sham')
            % Set pre to be first 10 minutes
            preInds = thisSamp<=600;
            % Set sham inds to be 1 hour after pre
            shamInds = thisSamp>600 & thisSamp<=4200;
            % Set post inds to be after sham
            postInds = thisSamp>4200;

            % Pre
            preData = thisData(preInds,:);
            preSamps = thisSamp(preInds);
            stimTime = preSamps-600;
            allData = [allData;table(preSamps',stimTime',...
                repmat(ratID,size(preData,1),1),...
                repmat({baseID},size(preData,1),1),...
                repmat({group},size(preData,1),1),...
                repmat(thisDate,size(preData,1),1),...
                repmat({'pre'},size(preData,1),1),preData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})];
            % Sham
            shamData = thisData(shamInds,:);
            shamSamps = thisSamp(shamInds);
            shamTime = shamSamps-600;
            allData = [allData;table(shamSamps',shamTime',...
                repmat(ratID,size(shamData,1),1),...
                repmat({baseID},size(shamData,1),1),...
                repmat({group},size(shamData,1),1),...
                repmat(thisDate,size(shamData,1),1),...
                repmat({'sham'},size(shamData,1),1),shamData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})];
            % Post
            postData = thisData(postInds,:);
            postSamps = thisSamp(postInds);
            stimTime = postSamps-4200;
            allData = [allData;table(postSamps',stimTime',...
                repmat(ratID,size(postData,1),1),...
                repmat({baseID},size(postData,1),1),...
                repmat({group},size(postData,1),1),...
                repmat(thisDate,size(postData,1),1),...
                repmat({'post'},size(postData,1),1),postData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})];
        else
            % Get baseline 'pre' data
            thisData = data{thisRat};
            thisSamp = samps{thisRat};
            thisDate = datetime(info{thisRat,2});
            % Add to time
            stimTime = NaN(height(thisData),1);
            allData = [allData;table(thisSamp',stimTime,...
                repmat(ratID,size(thisData,1),1),...
                repmat({baseID},size(thisData,1),1),...
                repmat({group},size(thisData,1),1),...
                repmat(thisDate,size(thisData,1),1),...
                repmat({'pre'},size(thisData,1),1),thisData,...
                'VariableNames',{'time','stimTime','fullID','ID','group',...
                'date','type','data'})];
        end
    end
end

% Normalize power
% Get starts and stops for normalizing power
normInds = [1:6:48; 6:6:48]';
% Set up empty table for all data
allData.normData = allData.data;
for ii = 1:size(allData,1)
    if mod(ii,1000) == 0
        disp(ii)
    end
    % Set up empty thisNorm vector
    thisNorm = zeros(1,48);
    for k = 1:size(normInds,1)
        thisSum = sum(allData.data(ii,normInds(k,1):normInds(k,2)));
        thisNorm(normInds(k,1):normInds(k,2)) = allData.data(ii,...
            normInds(k,1):normInds(k,2))./thisSum;
    end
    allData.normData(ii,1:48) = thisNorm;
end

% Set up simple ID vectors in the same order as the data
salData = allData(contains(allData.fullID, '-SAL-') & ...
    (strcmp(allData.type, 'pre') | (strcmp(allData.type, 'stim') & ...
    allData.stimTime <= 600)), :);
uSAL = unique(salData.fullID);
uShortSAL = cellfun(@(x) x{1},cellfun(@(x) strsplit(x,'-'),uSAL,...
    'uniformoutput',false),'uniformoutput',false);
lsdData = allData(contains(allData.fullID, '-LSD-') & ...
    (strcmp(allData.type, 'pre') | (strcmp(allData.type, 'stim') & ...
    allData.stimTime <= 600)), :);
uLSD = unique(lsdData.fullID);
uShortLSD = cellfun(@(x) x{1},cellfun(@(x) strsplit(x,'-'),uLSD,...
    'uniformoutput',false),'uniformoutput',false);
% Define rat IDs - 2 Hr stim
lsd_ids2 = uShortLSD';
sal_ids2 = uShortSAL';

% Save
save('H:\LSD+stim_persist\publicationData\cohort2Data.mat','allData', ...
    'data','files','samps','lsd_ids2','sal_ids2','uShortLSD','uShortSAL')
%% Model building
% Set number of permutations
nPerms = 10;

% Prepare cohort 1-3 data
load('H:\LSD+stim_persist\publicationData\cohort1-3Data.mat')
% Set washout times
times1 = {'24','48','72','144','168'};
% SAL
[preSAL1,stimSAL1,postSAL1] = deal(cell(numel(sal_ids1,1)));
washSAL1 = cell(numel(sal_ids1,numel(times1)));
% Separate data
for jj = 1:numel(sal_ids1)
    % Get Pre data
    preSAL1{jj} = allData(contains(allData.fullID,sal_ids1{jj}) & ...
        strcmp(allData.type,'pre') & contains(allData.group,'SAL'),:);
    % Get Stim data
    stimSAL1{jj} = allData(contains(allData.fullID,sal_ids1{jj}) & ...
        strcmp(allData.type,'stim') & contains(allData.group,'SAL'),:);
    % Get Post data
    postSAL1{jj} = allData(contains(allData.fullID,sal_ids1{jj}) & ...
        strcmp(allData.type,'post') & contains(allData.group,'SAL'),:);
    % Get Wash data
    for ii = 1:numel(times1)
        washSAL1{jj,ii} = allData(contains(allData.fullID,sal_ids1{jj}) & ...
            strcmp(allData.group,[times1{ii},'PostSAL']) & ...
            contains(allData.group,'SAL'),:);
    end
end
% LSD
[preLSD1,stimLSD1,postLSD1] = deal(cell(numel(lsd_ids1,1)));
washLSD1 = cell(numel(lsd_ids1,numel(times1)));
% Separate data
for jj = 1:numel(lsd_ids1)
    % Get Pre data
    preLSD1{jj} = allData(contains(allData.fullID,lsd_ids1{jj}) & ...
        strcmp(allData.type,'pre') & contains(allData.group,'LSD'),:);
    % Get Stim data
    stimLSD1{jj} = allData(contains(allData.fullID,lsd_ids1{jj}) & ...
        strcmp(allData.type,'stim') & contains(allData.group,'LSD'),:);
    % Get Post data
    postLSD1{jj} = allData(contains(allData.fullID,lsd_ids1{jj}) & ...
        strcmp(allData.type,'post') & contains(allData.group,'LSD'),:);
    % Get Wash data
    for ii = 1:numel(times1)
        washLSD1{jj,ii} = allData(contains(allData.fullID,lsd_ids1{jj}) & ...
            strcmp(allData.group,[times1{ii},'PostLSD']) & ...
            contains(allData.group,'LSD'),:);
    end
end
% Prepare cohort 2 data
load('H:\LSD+stim_persist\publicationData\cohort2Data.mat')
% Set washout times
times2 = {'24','48'};
% SAL
[preSAL2,stimSAL2,postSAL2] = deal(cell(numel(sal_ids2,1)));
washSAL2 = cell(numel(sal_ids2,numel(times2)));
% Separate data
for jj = 1:numel(sal_ids2)
    % Get Pre data
    preSAL2{jj} = allData(contains(allData.fullID,sal_ids2{jj}) & ...
        strcmp(allData.type,'pre') & contains(allData.group,'SAL'),:);
    % Get Stim data
    stimSAL2{jj} = allData(contains(allData.fullID,sal_ids2{jj}) & ...
        strcmp(allData.type,'stim') & contains(allData.group,'SAL'),:);
    % Get Post data
    postSAL2{jj} = allData(contains(allData.fullID,sal_ids2{jj}) & ...
        strcmp(allData.type,'post') & contains(allData.group,'SAL'),:);
    % Get Wash data
    for ii = 1:numel(times2)
        washSAL2{jj,ii} = allData(contains(allData.fullID,sal_ids2{jj}) & ...
            strcmp(allData.group,[times2{ii},'PostSAL']) & ...
            contains(allData.group,'SAL'),:);
    end
end
% LSD
[preLSD2,stimLSD2,postLSD2] = deal(cell(numel(lsd_ids2,1)));
washLSD2 = cell(numel(lsd_ids2,numel(times2)));
% Separate data
for jj = 1:numel(lsd_ids2)
    % Get Pre data
    preLSD2{jj} = allData(contains(allData.fullID,lsd_ids2{jj}) & ...
        strcmp(allData.type,'pre') & contains(allData.group,'LSD'),:);
    % Get Stim data
    stimLSD2{jj} = allData(contains(allData.fullID,lsd_ids2{jj}) & ...
        strcmp(allData.type,'stim') & contains(allData.group,'LSD'),:);
    % Get Post data
    postLSD2{jj} = allData(contains(allData.fullID,lsd_ids2{jj}) & ...
        strcmp(allData.type,'post') & contains(allData.group,'LSD'),:);
    % Get Wash data
    for ii = 1:numel(times2)
        washLSD2{jj,ii} = allData(contains(allData.fullID,lsd_ids2{jj}) & ...
            strcmp(allData.group,[times2{ii},'PostLSD']) & ...
            contains(allData.group,'LSD'),:);
    end
end

% Set up combined rat IDs (all cohorts)
sal_ids_all = [sal_ids2,sal_ids1];
lsd_ids_all = [lsd_ids2,lsd_ids1];

% Set male and female indices
% Load demog data
load('H:\LSD+stim_persist\publicationData\demogEphys.mat')
lsd_mIds = contains(demogEphys(matches(demogEphys(:,1),lsd_ids_all),2),'M');
lsd_fIds = contains(demogEphys(matches(demogEphys(:,1),lsd_ids_all),2),'F');
sal_mIds = contains(demogEphys(matches(demogEphys(:,1),sal_ids_all),2),'M');
sal_fIds = contains(demogEphys(matches(demogEphys(:,1),sal_ids_all),2),'F');

% Pre-allocate
% times1 x nPerms
[lsdA_glm80_comb1_wash,lsdA_glm80_comb1_wash_stim,salA_glm80_comb1_wash, ...
    salA_glm80_comb1_wash_stim,lsdA_glm80_1hr_wash,salA_glm80_1hr_wash, ...
    salA_glm80_comb1_wash_perm,lsdA_glm80_comb1_wash_perm, ...
    salA_glm80_1hr_wash_single,lsdA_glm80_1hr_wash_single] = ...
    deal(NaN(numel(times1),nPerms));
% times2 x nPerms
[lsdA_glm80_comb_wash,salA_glm80_comb_wash,lsdA_glm80_comb_wash_stim, ... 
    salA_glm80_comb_wash_stim,lsdA_glm80_2hr_wash,salA_glm80_2hr_wash, ...
    salA_glm80_2hr_wash_perm,lsdA_glm80_2hr_wash_perm, ...
    salA_glm80_comb_wash_perm,lsdA_glm80_comb_wash_perm, ...
    salA_glm80_comb_wash_single_perm,lsdA_glm80_comb_wash_single_perm] = ...
    deal(NaN(numel(times2,nPerms)));
% lsd_ids_all x bins (12) x nPerms
[lsdA_glm80_comb_postBinInd] = NaN(numel(lsd_ids_all),12,nPerms);
% sal_ids_all x bins (12) x nPerms
[salA_glm80_comb_postBinInd] = NaN(numel(sal_ids_all),12,nPerms);
% times2 x features (216) x nPerms 
[lsdA_glm80_comb_wash_single,lsdA_glm80_2hr_wash_single, ...
    salA_glm80_comb_wash_single,salA_glm80_2hr_wash_single, ...
    salA_glm80_comb_stim_wash_single,lsdA_glm80_comb_stim_wash_single] = deal(NaN(numel(times2),216,nPerms));
% features (216) x nPerms
[lsdA_glm80_comb_single,salA_glm80_comb_single,postSignLSDcombStim, ...
    lsdA_glm80_comb_single_stim,postSignSALcomb,postSignLSDcomb, ...
    postSignPermSALcomb,postSignPermLSDcomb,salA_glm80_comb_single_perm, ...
    lsdA_glm80_comb_single_perm,postSignSAL1,postSignSAL2, ... 
    postSignLSD1,postSignLSD2,salA_glm80_1hr_single, ...
    salA_glm80_2hr_single,lsdA_glm80_1hr_single,lsdA_glm80_2hr_single, ... 
    postSignPermSAL1,postSignPermSAL2,postSignPermLSD1,postSignPermLSD2, ...
    salA_glm80_1hr_single_perm,salA_glm80_2hr_single_perm, ... 
    lsdA_glm80_1hr_single_perm,lsdA_glm80_2hr_single_perm] = deal(NaN(216,nPerms));
% 1 x nPerms
[lsdA_glm80_comb_salTrain,salA_glm80_comb_lsdTrain, ...
    lsdVsalA_glm80_stim_comb,lsdVsalA_glm80_post_comb, ... 
    lsdVsalA_glm80_stim_comb_perm,lsdVsalA_glm80_post_comb_perm, ...
    lsdA_glm80_stim_comb,lsdA_glm80_comb,lsdA_glm80_comb_stim_post, ...
    salA_glm80_stim_comb,salA_glm80_comb,salA_glm80_comb_stim_post, ...
    lsdA_glm80_1hr_perm,salA_glm80_1hr_perm,lsdA_glm80_2hr, ... 
    lsdA_glm80_2hr_perm,salA_glm80_2hr_perm,lsdA_glm80_1hr_salTrain, ...
    salA_glm80_1hr_lsdTrain,lsdA_glm80_2hr_salTrain, ...
    salA_glm80_2hr_lsdTrain,lsdA_glm80_1hr_stim,lsdA_glm80_1hr, ...
    lsdA_glm80_1hr_stim_post,salA_glm80_1hr_stim,salA_glm80_1hr, ...
    salA_glm80_1hr_stim_post,lsdA_glm80_2hr_stim, ...
    lsdA_glm80_2hr_stim_post,salA_glm80_2hr_stim, ...
    salA_glm80_2hr,salA_glm80_2hr_stim_post,salA_glm80_comb_perm, ...
    lsdA_glm80_comb_perm,salA_glm80_comb_m,lsdA_glm80_comb_m, ...
    salA_glm80_comb_f,lsdA_glm80_comb_f] = deal(NaN(1,nPerms));
%
for k = 1:nPerms
    disp(k)
    %% Use up to 100 of all samples - 1 Hr (cohorts 1 and 3)
    % SAL
    [subPreSAL1,subPostSAL1,postIndSAL1,subStimSAL1,stimIndSAL1, ...
        subPostSAL1z,subStimSAL1z] = deal(cell(numel(sal_ids1),1));
    subWashSAL1 = cell(numel(sal_ids1),numel(times1));
    for jj = 1:numel(sal_ids1)
        nPre1  = min([height(preSAL1{jj}) 100]);
        nPost1 = min([height(postSAL1{jj}) 100]);
        nStim1 = min([height(stimSAL1{jj}) 100]);
        subPreSAL1{jj} = preSAL1{jj}.normData(randperm(height( ...
            preSAL1{jj}),nPre1),:);

        postIndSAL1{jj} = randperm(height(postSAL1{jj}),nPost1);
        subPostSAL1{jj} = postSAL1{jj}.normData(postIndSAL1{jj},:);
        subPostSAL1z{jj} = (postSAL1{jj}.data(postIndSAL1{jj},:) - ...
            mean(preSAL1{jj}.data))./std(preSAL1{jj}.data);

        stimIndSAL1{jj} = randperm(height(stimSAL1{jj}),nStim1);
        subStimSAL1{jj} = stimSAL1{jj}.normData(stimIndSAL1{jj},:);
        subStimSAL1z{jj} = (stimSAL1{jj}.data(stimIndSAL1{jj},:) - ...
            mean(preSAL1{jj}.data))./std(preSAL1{jj}.data);
        for ii = 1:numel(times1)
            nWash1 = min([height(washSAL1{jj,ii}) 100]);
            subWashSAL1{jj,ii} = washSAL1{jj,ii}.normData(randperm(...
                height(washSAL1{jj,ii}),nWash1),:);
        end
    end
    % LSD
    [subPreLSD1,subPostLSD1,postIndLSD1,subStimLSD1,stimIndLSD1,...
        subPostLSD1z,subStimLSD1z] = deal(cell(numel(lsd_ids1),1));
    subWashLSD1 = cell(numel(lsd_ids1),numel(times1));
    for jj = 1:numel(lsd_ids1)
        nPre1  = min([height(preLSD1{jj}) 100]);
        nPost1 = min([height(postLSD1{jj}) 100]);
        nStim1 = min([height(stimLSD1{jj}) 100]);
        subPreLSD1{jj} = preLSD1{jj}.normData(randperm(height( ...
            preLSD1{jj}),nPre1),:);

        postIndLSD1{jj} = randperm(height(postLSD1{jj}),nPost1);
        subPostLSD1{jj} = postLSD1{jj}.normData(postIndLSD1{jj},:);
        subPostLSD1z{jj} = (postLSD1{jj}.data(postIndLSD1{jj},:) - ...
            mean(preLSD1{jj}.data))./std(preLSD1{jj}.data);

        stimIndLSD1{jj} = randperm(height(stimLSD1{jj}),nStim1);
        subStimLSD1{jj} = stimLSD1{jj}.normData(stimIndLSD1{jj},:);
        subStimLSD1z{jj} = (stimLSD1{jj}.data(stimIndLSD1{jj},:) - ...
            mean(preLSD1{jj}.data))./std(preLSD1{jj}.data);
        for ii = 1:numel(times1)
            nWash1 = min([height(washLSD1{jj,ii}) 100]);
            subWashLSD1{jj,ii} = washLSD1{jj,ii}.normData(randperm( ...
                height(washLSD1{jj,ii}),nWash1),:);
        end
    end

    % split into 80:20 chunks
    % SAL
    [subPreTrainSAL1,subPostTrainSAL1,subPreTestSAL1,subPostTestSAL1, ...
        subStimTrainSAL1,subStimTestSAL1,subPostTrainSAL1z, ...
        subStimTrainSAL1z,subPostTestSAL1z,subStimTestSAL1z] = ...
        deal(cell(numel(sal_ids1),1));
    for jj = 1:numel(sal_ids1)
        preTrainIndSAL1 = randperm(height(subPreSAL1{jj}), ...
            round(height(subPreSAL1{jj}).*.8));
        preTestIndSAL1 = find(~ismember(1:height(subPreSAL1{jj}), ...
            preTrainIndSAL1));
        postTrainIndSAL1 = randperm(height(subPostSAL1{jj}), ...
            round(height(subPostSAL1{jj}).*.8));
        postTestIndSAL1 = find(~ismember(1:height(subPostSAL1{jj}), ...
            postTrainIndSAL1));
        stimTrainIndSAL1 =  randperm(height(subStimSAL1{jj}), ...
            round(height(subStimSAL1{jj}).*.8));
        stimTestIndSAL1 = find(~ismember(1:height(subStimSAL1{jj}), ...
            stimTrainIndSAL1));

        subPreTrainSAL1{jj} = subPreSAL1{jj}(preTrainIndSAL1,:);
        subPostTrainSAL1{jj} = subPostSAL1{jj}(postTrainIndSAL1,:);
        subStimTrainSAL1{jj} = subStimSAL1{jj}(stimTrainIndSAL1,:);
        subPostTrainSAL1z{jj} = subPostSAL1z{jj}(postTrainIndSAL1,:);
        subStimTrainSAL1z{jj} = subStimSAL1z{jj}(stimTrainIndSAL1,:);

        % Make sure test data has equal contributions from each rat
        subPreTestSAL1{jj} = subPreSAL1{jj}(preTestIndSAL1(randperm( ...
            numel(preTestIndSAL1), ...
            min([numel(preTestIndSAL1) numel(postTestIndSAL1)]))),:);
        subPostTestSAL1{jj} = subPostSAL1{jj}(postTestIndSAL1(randperm( ...
            numel(postTestIndSAL1), ...
            min([numel(preTestIndSAL1) numel(postTestIndSAL1)]))),:);
        subStimTestSAL1{jj} = subStimSAL1{jj}(stimTestIndSAL1(randperm( ...
            numel(stimTestIndSAL1), ...
            min([numel(preTestIndSAL1) numel(stimTestIndSAL1)]))),:);
        subPostTestSAL1z{jj} = subPostSAL1z{jj}(postTestIndSAL1(randperm( ...
            numel(postTestIndSAL1), ...
            min([numel(preTestIndSAL1) numel(postTestIndSAL1)]))),:);
        subStimTestSAL1z{jj} = subStimSAL1z{jj}(stimTestIndSAL1(randperm( ...
            numel(stimTestIndSAL1), ...
            min([numel(preTestIndSAL1) numel(stimTestIndSAL1)]))),:);
    end
    % LSD
    [subPreTrainLSD1,subPostTrainLSD1,subPreTestLSD1,subPostTestLSD1, ...
        subStimTrainLSD1,subStimTestLSD1,subPostTrainLSD1z, ...
        subStimTrainLSD1z,subPostTestLSD1z,subStimTestLSD1z] = ...
        deal(cell(numel(lsd_ids1),1));
    for jj = 1:numel(lsd_ids1)
        preTrainIndLSD1 = randperm(height(subPreLSD1{jj}), ...
            round(height(subPreLSD1{jj}).*.8));
        preTestIndLSD1 = find(~ismember(1:height(subPreLSD1{jj}), ...
            preTrainIndLSD1));
        postTrainIndLSD1 = randperm(height(subPostLSD1{jj}), ...
            round(height(subPostLSD1{jj}).*.8));
        postTestIndLSD1 = find(~ismember(1:height(subPostLSD1{jj}), ...
            postTrainIndLSD1));
        stimTrainIndLSD1 =  randperm(height(subStimLSD1{jj}), ...
            round(height(subStimLSD1{jj}).*.8));
        stimTestIndLSD1 = find(~ismember(1:height(subStimLSD1{jj}), ...
            stimTrainIndLSD1));

        subPreTrainLSD1{jj} = subPreLSD1{jj}(preTrainIndLSD1,:);
        subPostTrainLSD1{jj} = subPostLSD1{jj}(postTrainIndLSD1,:);
        subStimTrainLSD1{jj} = subStimLSD1{jj}(stimTrainIndLSD1,:);
        subPostTrainLSD1z{jj} = subPostLSD1z{jj}(postTrainIndLSD1,:);
        subStimTrainLSD1z{jj} = subStimLSD1z{jj}(stimTrainIndLSD1,:);
        % Make sure test data has equal contributions from each rat
        subPreTestLSD1{jj} = subPreLSD1{jj}(preTestIndLSD1(randperm( ...
            numel(preTestIndLSD1), ...
            min([numel(preTestIndLSD1) numel(postTestIndLSD1)]))),:);
        subPostTestLSD1{jj} = subPostLSD1{jj}(postTestIndLSD1(randperm( ...
            numel(postTestIndLSD1), ...
            min([numel(preTestIndLSD1) numel(postTestIndLSD1)]))),:);
        subStimTestLSD1{jj} = subStimLSD1{jj}(stimTestIndLSD1(randperm( ...
            numel(stimTestIndLSD1), ...
            min([numel(preTestIndLSD1) numel(stimTestIndLSD1)]))),:);
        subPostTestLSD1z{jj} = subPostLSD1z{jj}(postTestIndLSD1(randperm( ...
            numel(postTestIndLSD1), ...
            min([numel(preTestIndLSD1) numel(postTestIndLSD1)]))),:);
        subStimTestLSD1z{jj} = subStimLSD1z{jj}(stimTestIndLSD1(randperm( ...
            numel(stimTestIndLSD1), ...
            min([numel(preTestIndLSD1) numel(stimTestIndLSD1)]))),:);
    end
    % calculate weights
    % SAL
    nPre1 = cellfun(@height,subPreTrainSAL1);
    nPost1 = cellfun(@height,subPostTrainSAL1);
    nStim1 = cellfun(@height,subStimTrainSAL1);
    preSALW1 = cell(numel(sal_ids1),1);
    postSALW1 = cell(numel(sal_ids1),1);
    stimSALW1 = cell(numel(sal_ids1),1);
    for jj = 1:numel(sal_ids1)
        preSALW1{jj} = repmat((0.5/numel(sal_ids1))./nPre1(jj), ...
            nPre1(jj),1);
        postSALW1{jj} = repmat((0.5/numel(sal_ids1))./nPost1(jj), ...
            nPost1(jj),1);
        stimSALW1{jj} = repmat((0.5/numel(sal_ids1))./nStim1(jj), ...
            nStim1(jj),1);
    end

    % LSD
    nPre1 = cellfun(@height,subPreTrainLSD1);
    nPost1 = cellfun(@height,subPostTrainLSD1);
    nStim1 = cellfun(@height,subStimTrainLSD1);
    preLSDW1 = cell(numel(lsd_ids1),1);
    postLSDW1 = cell(numel(lsd_ids1),1);
    stimLSDW1 = cell(numel(lsd_ids1),1);
    for jj = 1:numel(lsd_ids1)
        preLSDW1{jj} = repmat((0.5/numel(lsd_ids1))./nPre1(jj), ...
            nPre1(jj),1);
        postLSDW1{jj} = repmat((0.5/numel(lsd_ids1))./nPost1(jj), ...
            nPost1(jj),1);
        stimLSDW1{jj} = repmat((0.5/numel(lsd_ids1))./nStim1(jj), ...
            nStim1(jj),1);
    end
    %% Use up to 100 of all samples - 2 Hr
    % SAL
    [subPreSAL2,subPostSAL2,postIndSAL2,subStimSAL2,stimIndSAL2, ...
        subPostSAL2z,subStimSAL2z] = deal(cell(numel(sal_ids2),1));
    subWashSAL2 = cell(numel(sal_ids2),numel(times2));
    for jj = 1:numel(sal_ids2)
        nPre2  = min([height(preSAL2{jj}) 100]);
        nPost2 = min([height(postSAL2{jj}) 100]);
        nStim2 = min([height(stimSAL2{jj}) 100]);
        subPreSAL2{jj} = preSAL2{jj}.normData(randperm(height(preSAL2{jj}), ...
            nPre2),:);

        postIndSAL2{jj} = randperm(height(postSAL2{jj}),nPost2);
        subPostSAL2{jj} = postSAL2{jj}.normData(postIndSAL2{jj},:);
        subPostSAL2z{jj} = (postSAL2{jj}.data(postIndSAL2{jj},:) - ...
            mean(preSAL2{jj}.data))./std(preSAL2{jj}.data);

        stimIndSAL2{jj} = randperm(height(stimSAL2{jj}),nStim2);
        subStimSAL2{jj} = stimSAL2{jj}.normData(stimIndSAL2{jj},:);
        subStimSAL2z{jj} = (stimSAL2{jj}.data(stimIndSAL2{jj},:) - ...
            mean(preSAL2{jj}.data))./std(preSAL2{jj}.data);
        for ii = 1:numel(times2)
            nWash = min([height(washSAL2{jj,ii}) 100]);
            subWashSAL2{jj,ii} = washSAL2{jj,ii}.normData(randperm( ...
                height(washSAL2{jj,ii}),nWash),:);
        end
    end
    % LSD
    [subPreLSD2,subPostLSD2,postIndLSD2,subStimLSD2,stimIndLSD2, ...
        subPostLSD2z,subStimLSD2z] = deal(cell(numel(lsd_ids2),1));
    subWashLSD2 = cell(numel(lsd_ids2),numel(times2));
    for jj = 1:numel(lsd_ids2)
        nPre2  = min([height(preLSD2{jj}) 100]);
        nPost2 = min([height(postLSD2{jj}) 100]);
        nStim2 = min([height(stimLSD2{jj}) 100]);
        subPreLSD2{jj} = preLSD2{jj}.normData(randperm(height(preLSD2{jj}), ...
            nPre2),:);

        postIndLSD2{jj} = randperm(height(postLSD2{jj}),nPost2);
        subPostLSD2{jj} = postLSD2{jj}.normData(postIndLSD2{jj},:);
        subPostLSD2z{jj} = (postLSD2{jj}.data(postIndLSD2{jj},:) - ...
            mean(preLSD2{jj}.data))./std(preLSD2{jj}.data);

        stimIndLSD2{jj} = randperm(height(stimLSD2{jj}),nStim2);
        subStimLSD2{jj} = stimLSD2{jj}.normData(stimIndLSD2{jj},:);
        subStimLSD2z{jj} = (stimLSD2{jj}.data(stimIndLSD2{jj},:) - ...
            mean(preLSD2{jj}.data))./std(preLSD2{jj}.data);
        for ii = 1:numel(times2)
            nWash = min([height(washLSD2{jj,ii}) 100]);
            subWashLSD2{jj,ii} = washLSD2{jj,ii}.normData(randperm( ...
                height(washLSD2{jj,ii}),nWash),:);
        end
    end
    % split into 80:20 chunks
    % SAL
    [subPreTrainSAL2,subPostTrainSAL2,subPreTestSAL2,subPostTestSAL2, ...
        subStimTrainSAL2,subStimTestSAL2,subPostTrainSAL2z, ...
        subStimTrainSAL2z,subPostTestSAL2z,subStimTestSAL2z] = ...
        deal(cell(numel(sal_ids2),1));
    for jj = 1:numel(sal_ids2)
        preTrainIndSAL2 = randperm(height(subPreSAL2{jj}), ...
            round(height(subPreSAL2{jj}).*.8));
        preTestIndSAL2 = find(~ismember(1:height(subPreSAL2{jj}), ...
            preTrainIndSAL2));
        postTrainIndSAL2 = randperm(height(subPostSAL2{jj}), ...
            round(height(subPostSAL2{jj}).*.8));
        postTestIndSAL2 = find(~ismember(1:height(subPostSAL2{jj}), ...
            postTrainIndSAL2));
        stimTrainIndSAL2 =  randperm(height(subStimSAL2{jj}), ...
            round(height(subStimSAL2{jj}).*.8));
        stimTestIndSAL2 = find(~ismember(1:height(subStimSAL2{jj}), ...
            stimTrainIndSAL2));

        subPreTrainSAL2{jj} = subPreSAL2{jj}(preTrainIndSAL2,:);
        subPostTrainSAL2{jj} = subPostSAL2{jj}(postTrainIndSAL2,:);
        subStimTrainSAL2{jj} = subStimSAL2{jj}(stimTrainIndSAL2,:);
        subPostTrainSAL2z{jj} = subPostSAL2z{jj}(postTrainIndSAL2,:);
        subStimTrainSAL2z{jj} = subStimSAL2z{jj}(stimTrainIndSAL2,:);

        % Make sure test data has equal contributions from each rat
        subPreTestSAL2{jj} = subPreSAL2{jj}(preTestIndSAL2(randperm(numel( ...
            preTestIndSAL2), ...
            min([numel(preTestIndSAL2) numel(postTestIndSAL2)]))),:);
        subPostTestSAL2{jj} = subPostSAL2{jj}(postTestIndSAL2(randperm( ...
            numel(postTestIndSAL2), ...
            min([numel(preTestIndSAL2) numel(postTestIndSAL2)]))),:);
        subStimTestSAL2{jj} = subStimSAL2{jj}(stimTestIndSAL2(randperm( ...
            numel(stimTestIndSAL2), ...
            min([numel(preTestIndSAL2) numel(stimTestIndSAL2)]))),:);
        subPostTestSAL2z{jj} = subPostSAL2z{jj}(postTestIndSAL2(randperm( ...
            numel(postTestIndSAL2), ...
            min([numel(preTestIndSAL2) numel(postTestIndSAL2)]))),:);
        subStimTestSAL2z{jj} = subStimSAL2z{jj}(stimTestIndSAL2(randperm( ...
            numel(stimTestIndSAL2), ...
            min([numel(preTestIndSAL2) numel(stimTestIndSAL2)]))),:);
    end
    % LSD
    [subPreTrainLSD2,subPostTrainLSD2,subPreTestLSD2,subPostTestLSD2, ...
        subStimTrainLSD2,subStimTestLSD2,subPostTrainLSD2z, ...
        subStimTrainLSD2z,subPostTestLSD2z,subStimTestLSD2z] = ...
        deal(cell(numel(lsd_ids2),1));
    for jj = 1:numel(lsd_ids2)
        preTrainIndLSD2 = randperm(height(subPreLSD2{jj}), ...
            round(height(subPreLSD2{jj}).*.8));
        preTestIndLSD2 = find(~ismember(1:height(subPreLSD2{jj}), ...
            preTrainIndLSD2));
        postTrainIndLSD2 = randperm(height(subPostLSD2{jj}), ...
            round(height(subPostLSD2{jj}).*.8));
        postTestIndLSD2 = find(~ismember(1:height(subPostLSD2{jj}), ...
            postTrainIndLSD2));
        stimTrainIndLSD2 =  randperm(height(subStimLSD2{jj}), ...
            round(height(subStimLSD2{jj}).*.8));
        stimTestIndLSD2 = find(~ismember(1:height(subStimLSD2{jj}), ...
            stimTrainIndLSD2));

        subPreTrainLSD2{jj} = subPreLSD2{jj}(preTrainIndLSD2,:);
        subPostTrainLSD2{jj} = subPostLSD2{jj}(postTrainIndLSD2,:);
        subStimTrainLSD2{jj} = subStimLSD2{jj}(stimTrainIndLSD2,:);
        subPostTrainLSD2z{jj} = subPostLSD2z{jj}(postTrainIndLSD2,:);
        subStimTrainLSD2z{jj} = subStimLSD2z{jj}(stimTrainIndLSD2,:);

        % Make sure test data has equal contributions from each rat
        subPreTestLSD2{jj} = subPreLSD2{jj}(preTestIndLSD2(randperm(numel( ...
            preTestIndLSD2), ...
            min([numel(preTestIndLSD2) numel(postTestIndLSD2)]))),:);
        subPostTestLSD2{jj} = subPostLSD2{jj}(postTestIndLSD2(randperm( ...
            numel(postTestIndLSD2), ...
            min([numel(preTestIndLSD2) numel(postTestIndLSD2)]))),:);
        subStimTestLSD2{jj} = subStimLSD2{jj}(stimTestIndLSD2(randperm( ...
            numel(stimTestIndLSD2), ...
            min([numel(preTestIndLSD2) numel(stimTestIndLSD2)]))),:);
        subPostTestLSD2z{jj} = subPostLSD2z{jj}(postTestIndLSD2(randperm( ...
            numel(postTestIndLSD2), ...
            min([numel(preTestIndLSD2) numel(postTestIndLSD2)]))),:);
        subStimTestLSD2z{jj} = subStimLSD2z{jj}(stimTestIndLSD2(randperm( ...
            numel(stimTestIndLSD2), ...
            min([numel(preTestIndLSD2) numel(stimTestIndLSD2)]))),:);
    end
    % calculate weights
    % SAL
    nPre2 = cellfun(@height,subPreTrainSAL2);
    nPost2 = cellfun(@height,subPostTrainSAL2);
    nStim2 = cellfun(@height,subStimTrainSAL2);
    preSALW2 = cell(numel(sal_ids2),1);
    postSALW2 = cell(numel(sal_ids2),1);
    stimSALW2 = cell(numel(sal_ids2),1);
    for jj = 1:numel(sal_ids2)
        preSALW2{jj} = repmat((0.5/numel(sal_ids2))./nPre2(jj), ...
            nPre2(jj),1);
        postSALW2{jj} = repmat((0.5/numel(sal_ids2))./nPost2(jj), ...
            nPost2(jj),1);
        stimSALW2{jj} = repmat((0.5/numel(sal_ids2))./nStim2(jj), ...
            nStim2(jj),1);
    end

    % LSD
    nPre2 = cellfun(@height,subPreTrainLSD2);
    nPost2 = cellfun(@height,subPostTrainLSD2);
    nStim2 = cellfun(@height,subStimTrainLSD2);
    preLSDW2 = cell(numel(lsd_ids2),1);
    postLSDW2 = cell(numel(lsd_ids2),1);
    stimLSDW2 = cell(numel(lsd_ids2),1);
    for jj = 1:numel(lsd_ids2)
        preLSDW2{jj} = repmat((0.5/numel(lsd_ids2))./nPre2(jj), ...
            nPre2(jj),1);
        postLSDW2{jj} = repmat((0.5/numel(lsd_ids2))./nPost2(jj), ...
            nPost2(jj),1);
        stimLSDW2{jj} = repmat((0.5/numel(lsd_ids2))./nStim2(jj), ...
            nStim2(jj),1);
    end
    %% LSD vs. SAL - combined cohorts - post (Figure 3E)
    % Grab real data
    trainXpostLSDvSAL = cat(1,subPostTrainSAL2z{:},subPostTrainSAL1z{:}, ...
        subPostTrainLSD2z{:},subPostTrainLSD1z{:});
    trainYpostLSDvSAL = [zeros(height(cat(1,subPostTrainSAL2z{:}, ...
        subPostTrainSAL1z{:})),1);ones(height(cat(1,subPostTrainLSD2z{:}, ...
        subPostTrainLSD1z{:})),1)];
    testXpostLSDvSAL = cat(1,subPostTestSAL2z{:},subPostTestSAL1z{:}, ...
        subPostTestLSD2z{:},subPostTestLSD1z{:});
    testYpostLSDvSAL = [zeros(height(cat(1,subPostTestSAL2z{:}, ...
        subPostTestSAL1z{:})),1);ones(height(cat(1,subPostTestLSD2z{:}, ...
        subPostTestLSD1z{:})),1)];
    weightsLSDvSALpost = cat(1,postSALW2{:},postSALW1{:},postLSDW2{:}, ...
        postLSDW1{:});
    % Build real model
    mdlPostLSDvSAL = fitglm(trainXpostLSDvSAL,trainYpostLSDvSAL, ...
        'distribution','binomial','weights',weightsLSDvSALpost);
    % Test real model
    scores = predict(mdlPostLSDvSAL,testXpostLSDvSAL);
    [~,~,~,lsdVsalA_glm80_post_comb(k)] = perfcurve(testYpostLSDvSAL, ...
        scores,1);

    % Permute data
    permInd = randperm(numel(trainYpostLSDvSAL),numel(trainYpostLSDvSAL));
    permTrainYpostLSDvSAL = trainYpostLSDvSAL(permInd);
    permWeightsLSDvSALpost = weightsLSDvSALpost(permInd);
    % Build permuted model
    mdlPermPostLSDvSAL = fitglm(trainXpostLSDvSAL,permTrainYpostLSDvSAL, ...
        'distribution','binomial','weights',permWeightsLSDvSALpost);
    % Test permuted model
    scores = predict(mdlPermPostLSDvSAL,testXpostLSDvSAL);
    [~,~,~,lsdVsalA_glm80_post_comb_perm(k)] = perfcurve(testYpostLSDvSAL, ...
        scores,1);
    %% LSD vs. SAL - combined cohorts - stim (Figure 3E)
    % Grab real data
    trainXstimLSDvSAL = cat(1,subStimTrainSAL2z{:},subStimTrainSAL1z{:}, ...
        subStimTrainLSD2z{:},subStimTrainLSD1z{:});
    trainYstimLSDvSAL = [zeros(height(cat(1,subStimTrainSAL2z{:}, ...
        subStimTrainSAL1z{:})),1);ones(height(cat(1,subStimTrainLSD2z{:}, ...
        subStimTrainLSD1z{:})),1)];
    testXstimLSDvSAL = cat(1,subStimTestSAL2z{:},subStimTestSAL1z{:}, ...
        subStimTestLSD2z{:},subStimTestLSD1z{:});
    testYstimLSDvSAL = [zeros(height(cat(1,subStimTestSAL2z{:}, ...
        subStimTestSAL1z{:})),1);ones(height(cat(1,subStimTestLSD2z{:}, ...
        subStimTestLSD1z{:})),1)];
    weightsLSDvSALstim = cat(1,stimSALW2{:},stimSALW1{:},stimLSDW2{:}, ...
        stimLSDW1{:});
    % Build real model
    mdlStimLSDvSAL = fitglm(trainXstimLSDvSAL,trainYstimLSDvSAL, ...
        'distribution','binomial','weights',weightsLSDvSALstim);
    % Test real model
    scores = predict(mdlStimLSDvSAL,testXstimLSDvSAL);
    [~,~,~,lsdVsalA_glm80_stim_comb(k)] = perfcurve(testYstimLSDvSAL, ...
        scores,1);

    % Permute data
    permInd = randperm(numel(trainYstimLSDvSAL),numel(trainYstimLSDvSAL));
    permTrainYstimLSDvSAL = trainYstimLSDvSAL(permInd);
    permWeightsLSDvSALstim = weightsLSDvSALstim(permInd);
    % Build permuted model
    mdlPermStimLSDvSAL = fitglm(trainXstimLSDvSAL,permTrainYstimLSDvSAL, ...
        'distribution','binomial','weights',permWeightsLSDvSALstim);
    % Test permuted model
    scores = predict(mdlPermStimLSDvSAL,testXstimLSDvSAL);
    [~,~,~,lsdVsalA_glm80_stim_comb_perm(k)] = perfcurve(testYstimLSDvSAL, ...
        scores,1);
    %% Pre vs. post - combined cohorts (Figure 2A)
    % Prepare SAL data
    trainXsalComb = cat(1,subPostTrainSAL2{:},subPreTrainSAL2{:}, ...
        subPostTrainSAL1{:},subPreTrainSAL1{:});
    trainYsalComb = [ones(height(cat(1,subPostTrainSAL2{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL2{:})),1); ...
        ones(height(cat(1,subPostTrainSAL1{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL1{:})),1)];
    testXsalComb = cat(1,subPostTestSAL2{:},subPreTestSAL2{:}, ...
        subPostTestSAL1{:},subPreTestSAL1{:});
    testYsalComb = [ones(height(cat(1,subPostTestSAL2{:})),1); ...
        zeros(height(cat(1,subPreTestSAL2{:})),1); ...
        ones(height(cat(1,subPostTestSAL1{:})),1); ...
        zeros(height(cat(1,subPreTestSAL1{:})),1)];
    % Get real SAL weights
    weightSALcomb = cat(1,postSALW2{:},preSALW2{:},postSALW1{:},preSALW1{:});
    % Build real SAL model
    salMdlComb = fitglm(trainXsalComb,trainYsalComb,'distribution', ...
        'binomial','weights',weightSALcomb);
    % Test real SAL model
    scores = predict(salMdlComb,testXsalComb);
    % Get real SAL model AUCs
    [~,~,~,salA_glm80_comb(k)] = perfcurve(testYsalComb,scores,1);
    
    % Permute SAL data
    permIndComb = randperm(numel(trainYsalComb),numel(trainYsalComb));
    permTrainYSALcomb = trainYsalComb(permIndComb);
    permWeightSALcomb = weightSALcomb(permIndComb);
    % Build permuted SAL model
    salMdlPermComb = fitglm(trainXsalComb,permTrainYSALcomb, ...
        'distribution','binomial','weights', permWeightSALcomb);
    % Test permuted SAL model
    scores = predict(salMdlPermComb,testXsalComb);
    % Get permuted SAL model AUCs
    [~,~,~,salA_glm80_comb_perm(k)] = perfcurve(testYsalComb,scores,1);
    
    % % Break down accuracy for each SAL rat
    % subPostTestSALcomb = [subPostTestSAL2;subPostTestSAL1];
    % subPreTestSALcomb = [subPreTestSAL2;subPreTestSAL1];
    % for jj = 1:numel(sal_ids_all)
    %     if ~isempty(subPostTestSALcomb{jj}) && ~isempty( ...
    %             subPreTestSALcomb{jj})
    %     scores = predict(salMdlComb,[subPostTestSALcomb{jj}; ...
    %         subPreTestSALcomb{jj}]);
    %     [~,~,~,salA_glm80_comb_ind(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestSALcomb{jj}),1); ...
    %         zeros(height(subPreTestSALcomb{jj}),1)],scores, 1);
    %     scores = predict(salMdlPermComb,[subPostTestSALcomb{jj}; ...
    %         subPreTestSALcomb{jj}]);
    %     [~,~,~,salA_glm80_comb_ind_perm(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestSALcomb{jj}),1); ...
    %         zeros(height(subPreTestSALcomb{jj}),1)],scores, 1);
    %     else
    %         salA_glm80_comb_ind(jj,k) = NaN;
    %         salA_glm80_comb_ind_perm(jj,k) = NaN;
    %     end
    % end
   
    % Prepare LSD data 
    trainXlsdComb = cat(1,subPostTrainLSD2{:},subPreTrainLSD2{:}, ...
        subPostTrainLSD1{:},subPreTrainLSD1{:});
    trainYlsdComb = [ones(height(cat(1,subPostTrainLSD2{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD2{:})),1); ...
        ones(height(cat(1,subPostTrainLSD1{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD1{:})),1)];
    testXlsdComb = cat(1,subPostTestLSD2{:},subPreTestLSD2{:}, ...
        subPostTestLSD1{:},subPreTestLSD1{:});
    testYlsdComb = [ones(height(cat(1,subPostTestLSD2{:})),1); ...
        zeros(height(cat(1,subPreTestLSD2{:})),1); ...
        ones(height(cat(1,subPostTestLSD1{:})),1); ...
        zeros(height(cat(1,subPreTestLSD1{:})),1)];
    % Get real LSD weights
    weightLSDcomb = cat(1,postLSDW2{:},preLSDW2{:},postLSDW1{:},preLSDW1{:});
    % Build real LSD model
    lsdMdlComb = fitglm(trainXlsdComb,trainYlsdComb,'distribution', ...
        'binomial','weights',weightLSDcomb);
    % Test real LSD model
    scores = predict(lsdMdlComb,testXlsdComb);
    % Get real LSD model AUCs
    [~,~,~,lsdA_glm80_comb(k)] = perfcurve(testYlsdComb,scores,1);
    
    % Permute LSD data
    permIndComb = randperm(numel(trainYlsdComb),numel(trainYlsdComb));
    permTrainYLSDcomb = trainYlsdComb(permIndComb);
    permWeightLSDcomb = weightLSDcomb(permIndComb);
    % Build permuted LSD model
    lsdMdlPermComb = fitglm(trainXlsdComb,permTrainYLSDcomb, ...
        'distribution','binomial','weights',permWeightLSDcomb);
    % Test permuted LSD model
    scores = predict(lsdMdlPermComb,testXlsdComb);
    % Get permuted LSD model AUCs
    [~,~,~,lsdA_glm80_comb_perm(k)] = perfcurve(testYlsdComb,scores,1);
    
    % % Break down accuracy for each LSD rat
    % subPostTestLSDcomb = [subPostTestLSD2;subPostTestLSD1];
    % subPreTestLSDcomb = [subPreTestLSD2;subPreTestLSD1];
    % for jj = 1:numel(lsd_ids_all)
    %     if ~isempty(subPostTestLSDcomb{jj}) && ...
    %         ~isempty(subPreTestLSDcomb{jj})
    %     scores = predict(lsdMdlComb,[subPostTestLSDcomb{jj}; ...
    %         subPreTestLSDcomb{jj}]);
    %     [~,~,~,lsdA_glm80_comb_ind(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestLSDcomb{jj}),1); ...
    %         zeros(height(subPreTestLSDcomb{jj}),1)],scores, 1);
    %     scores = predict(lsdMdlPermComb,[subPostTestLSDcomb{jj}; ...
    %         subPreTestLSDcomb{jj}]);
    %     [~,~,~,lsdA_glm80_comb_ind_perm(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestLSDcomb{jj}),1); ...
    %         zeros(height(subPreTestLSDcomb{jj}),1)],scores, 1);
    %     else
    %         lsdA_glm80_comb_ind(jj,k) = NaN;
    %         lsdA_glm80_comb_ind_perm(jj,k) = NaN;
    %     end
    % end
    %% Pre vs. post single features - combined cohorts
    % Preallocate
    [salMdlSingleComb,salMdlSinglePermComb,lsdMdlSingleComb, ...
        lsdMdlSinglePermComb] = deal(cell(1,216));
    for f = 1:216
        % Build real SAL single feature model
        salMdlSingleComb{f} = fitglm(trainXsalComb(:,f),trainYsalComb,...
            'distribution','binomial','weights',weightSALcomb);
        % Extract sign of beta coefficient from real SAL single feature
        % model
        postSignSALcomb(f,k) = sign(table2array( ...
            salMdlSingleComb{f}.Coefficients(2,1)));
        % Test real SAL single feature model
        scores = predict(salMdlSingleComb{f},testXsalComb(:,f));
        % Get real SAL single feature model AUCs
        [~,~,~,salA_glm80_comb_single(f,k)] = perfcurve(testYsalComb, ...
            scores,1);
        
        % Build permuted SAL single feature model
        salMdlSinglePermComb{f} = fitglm(trainXsalComb(:,f), ...
            permTrainYSALcomb,'distribution','binomial','weights', ...
            permWeightSALcomb);
        % Extract sign of beta coefficient from permuted SAL single feature
        % model
        postSignPermSALcomb(f,k) = sign(table2array( ...
            salMdlSinglePermComb{f}.Coefficients(2,1)));
        % Test permuted SAL single feature model
        scores = predict(salMdlSinglePermComb{f},testXsalComb(:,f));
        % Get permuted SAL single feature model AUCs
        [~,~,~,salA_glm80_comb_single_perm(f,k)] = perfcurve(testYsalComb, ...
            scores,1);

        % Build real LSD single feature model
        lsdMdlSingleComb{f} = fitglm(trainXlsdComb(:,f),trainYlsdComb, ...
            'distribution','binomial','weights',weightLSDcomb);
        % Extract sign of beta coefficient from real LSD single feature
        % model
        postSignLSDcomb(f,k) = sign(table2array( ...
            lsdMdlSingleComb{f}.Coefficients(2,1)));
        % Test real LSD single feature model
        scores = predict(lsdMdlSingleComb{f},testXlsdComb(:,f));
        % Get real LSD single feature model AUCs
        [~,~,~,lsdA_glm80_comb_single(f,k)] = perfcurve(testYlsdComb, ...
            scores,1);

        % Build permuted LSD single feature model
        lsdMdlSinglePermComb{f} = fitglm(trainXlsdComb(:,f), ...
            permTrainYLSDcomb,'distribution','binomial','weights', ...
            permWeightLSDcomb);
        % Extract sign of beta coefficient from permuted LSD single feature
        % model
        postSignPermLSDcomb(f,k) = sign(table2array( ...
            lsdMdlSinglePermComb{f}.Coefficients(2,1)));
        % Test permuted LSD single feature model
        scores = predict(lsdMdlSinglePermComb{f},testXlsdComb(:,f));
        % Extract sign of beta coefficient from permuted LSD single feature
        % model
        [~,~,~,lsdA_glm80_comb_single_perm(f,k)] = perfcurve(testYlsdComb, ...
            scores,1);
    end
    %% Pre vs. post split into male and female - combined cohorts (Extended
    %  Data Figure 2)
    % Prepare SAL data
    allPostTrainSAL = cat(1,subPostTrainSAL2(:),subPostTrainSAL1(:));
    allPreTrainSAL = cat(1,subPreTrainSAL2(:),subPreTrainSAL1(:));
    allPostTestSAL = cat(1,subPostTestSAL2(:),subPostTestSAL1(:));
    allPreTestSAL = cat(1,subPreTestSAL2(:),subPreTestSAL1(:));
    allPostWsal = cat(1,postSALW2(:),postSALW1(:));
    allPreWsal = cat(1,preSALW2(:),preSALW1(:));

    % Build SAL male model
    salMdlCombM = fitglm(cat(1,allPostTrainSAL{sal_mIds}, ...
        allPreTrainSAL{sal_mIds}),cat(1, ...
        ones(height(cat(1,allPostTrainSAL{sal_mIds})),1),...
        zeros(height(cat(1,allPreTrainSAL{sal_mIds})),1)),'distribution', ...
        'binomial','weights', ...
        cat(1,allPostWsal{sal_mIds},allPreWsal{sal_mIds}));
    % Test SAL male model
    scores = predict(salMdlCombM,cat(1,allPostTestSAL{sal_mIds}, ...
        allPreTestSAL{sal_mIds}));
    % Get SAL male AUCs
    [~,~,~,salA_glm80_comb_m(k)] = perfcurve(cat(1, ...
        ones(height(cat(1,allPostTestSAL{sal_mIds})),1), ...
        zeros(height(cat(1,allPreTestSAL{sal_mIds})),1)),scores,1);
    
    % Build SAL female model
    salMdlCombF = fitglm(cat(1,allPostTrainSAL{sal_fIds}, ...
        allPreTrainSAL{sal_fIds}),cat(1, ...
        ones(height(cat(1,allPostTrainSAL{sal_fIds})),1), ...
        zeros(height(cat(1,allPreTrainSAL{sal_fIds})),1)),'distribution', ...
        'binomial','weights', ...
        cat(1,allPostWsal{sal_fIds},allPreWsal{sal_fIds}));
    % Test SAL female model
    scores = predict(salMdlCombF,cat(1,allPostTestSAL{sal_fIds}, ...
        allPreTestSAL{sal_fIds}));
    % Get SAL female AUCs
    [~,~,~,salA_glm80_comb_f(k)] = perfcurve(cat(1, ...
        ones(height(cat(1,allPostTestSAL{sal_fIds})),1), ...
        zeros(height(cat(1,allPreTestSAL{sal_fIds})),1)),scores,1);
    
    % Prepare LSD data
    allPostTrainLSD = cat(1,subPostTrainLSD2(:),subPostTrainLSD1(:));
    allPreTrainLSD = cat(1,subPreTrainLSD2(:),subPreTrainLSD1(:));
    allPostTestLSD = cat(1,subPostTestLSD2(:),subPostTestLSD1(:));
    allPreTestLSD = cat(1,subPreTestLSD2(:),subPreTestLSD1(:));
    allPostWlsd = cat(1,postLSDW2(:),postLSDW1(:));
    allPreWlsd = cat(1,preLSDW2(:),preLSDW1(:));
    % Build LSD male model
    lsdMdlCombM = fitglm(cat(1,allPostTrainLSD{lsd_mIds}, ...
        allPreTrainLSD{lsd_mIds}),cat(1, ...
        ones(height(cat(1,allPostTrainLSD{lsd_mIds})),1), ...
        zeros(height(cat(1,allPreTrainLSD{lsd_mIds})),1)),'distribution', ...
        'binomial','weights', ...
        cat(1,allPostWlsd{lsd_mIds},allPreWlsd{lsd_mIds}));
    % Test LSD male model
    scores = predict(lsdMdlCombM,cat(1,allPostTestLSD{lsd_mIds}, ...
        allPreTestLSD{lsd_mIds}));
    % Get LSD male AUCs
    [~,~,~,lsdA_glm80_comb_m(k)] = perfcurve(cat(1, ...
        ones(height(cat(1,allPostTestLSD{lsd_mIds})),1), ...
        zeros(height(cat(1,allPreTestLSD{lsd_mIds})),1)),scores,1);
    % Build LSD female model
    lsdMdlCombF = fitglm(cat(1,allPostTrainLSD{lsd_fIds}, ...
        allPreTrainLSD{lsd_fIds}),cat(1, ...
        ones(height(cat(1,allPostTrainLSD{lsd_fIds})),1), ...
        zeros(height(cat(1,allPreTrainLSD{lsd_fIds})),1)),'distribution', ...
        'binomial','weights', ...
        cat(1,allPostWlsd{lsd_fIds},allPreWlsd{lsd_fIds}));
    % Test LSD female model
    scores = predict(lsdMdlCombF,cat(1,allPostTestLSD{lsd_fIds}, ...
        allPreTestLSD{lsd_fIds}));
    % Get LSD female AUCs
    [~,~,~,lsdA_glm80_comb_f(k)] = perfcurve(cat(1, ...
        ones(height(cat(1,allPostTestLSD{lsd_fIds})),1), ...
        zeros(height(cat(1,allPreTestLSD{lsd_fIds})),1)),scores,1);
    %% Pre vs. post - 1 hour stim (Extended Data Figure 4A)
    % Prepare SAL data
    trainXsal1 = cat(1,subPostTrainSAL1{:},subPreTrainSAL1{:});
    trainYsal1 = [ones(height(cat(1,subPostTrainSAL1{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL1{:})),1)];
    testXsal1 = cat(1,subPostTestSAL1{:},subPreTestSAL1{:});
    testYsal1 = [ones(height(cat(1,subPostTestSAL1{:})),1); ...
        zeros(height(cat(1,subPreTestSAL1{:})),1)];
    % Get SAL weights
    weightSAL1 = cat(1,postSALW1{:},preSALW1{:});
    % Build SAL model
    salMdl1 = fitglm(trainXsal1,trainYsal1,'distribution','binomial', ...
        'weights',weightSAL1);
    % Test SAL model
    scores = predict(salMdl1,testXsal1);
    % Get real SAL model AUCs
    [~,~,~,salA_glm80_1hr(k)] = perfcurve(testYsal1,scores,1);
    % Permuted SAL data
    permInd = randperm(numel(trainYsal1),numel(trainYsal1));
    permTrainYSAL1 = trainYsal1(permInd);
    permWeightSAL1 = weightSAL1(permInd);
    % Build permuted SAL model
    salMdlPerm1 = fitglm(trainXsal1,permTrainYSAL1,'distribution', ...
        'binomial','weights',permWeightSAL1);
    % Test permuted SAL model
    scores = predict(salMdlPerm1,testXsal1);
    % Get permuted SAL model AUCs
    [~,~,~,salA_glm80_1hr_perm(k)] = perfcurve(testYsal1,scores,1);
    % % Break down accuracy for each rat
    % for jj = 1:numel(sal_ids1)
    %     if ~isempty(subPostTestSAL1{jj}) && ~isempty(subPreTestSAL1{jj})
    %         scores = predict(salMdl1,[subPostTestSAL1{jj}; ...
    %             subPreTestSAL1{jj}]);
    %         [~,~,~,salA_glm80_1hr_ind(jj,k)]  = perfcurve([ones(height( ...
    %             subPostTestSAL1{jj}),1);...
    %             zeros(height(subPreTestSAL1{jj}),1)],scores, 1);
    %         scores = predict(salMdlPerm1,[subPostTestSAL1{jj}; ...
    %             subPreTestSAL1{jj}]);
    %         [~,~,~,salA_glm80_1hr_ind_perm(jj,k)]  = perfcurve([ones( ...
    %             height(subPostTestSAL1{jj}),1); ...
    %             zeros(height(subPreTestSAL1{jj}),1)],scores, 1);
    %     else
    %         salA_glm80_1hr_ind(jj,k) = NaN;
    %         salA_glm80_1hr_ind_perm(jj,k) = NaN;
    %     end
    % end

    % Prepare real LSD data
    trainXlsd1 = cat(1,subPostTrainLSD1{:},subPreTrainLSD1{:});
    trainYlsd1 = [ones(height(cat(1,subPostTrainLSD1{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD1{:})),1)];
    testXlsd1 = cat(1,subPostTestLSD1{:},subPreTestLSD1{:});
    testYlsd1 = [ones(height(cat(1,subPostTestLSD1{:})),1); ...
        zeros(height(cat(1,subPreTestLSD1{:})),1)];
    % Get real LSD weights
    weightLSD1 = cat(1,postLSDW1{:},preLSDW1{:});
    % Build real LSD model
    lsdMdl1 = fitglm(trainXlsd1,trainYlsd1,'distribution','binomial', ...
        'weights',weightLSD1);
    % Test real LSD model
    scores = predict(lsdMdl1,testXlsd1);
    % Get real LSD model AUCs
    [~,~,~,lsdA_glm80_1hr(k)] = perfcurve(testYlsd1,scores,1);
    % Permute LSD data
    permInd = randperm(numel(trainYlsd1),numel(trainYlsd1));
    permTrainYLSD1 = trainYlsd1(permInd);
    permWeightLSD1 = weightLSD1(permInd);
    % Build permuted LSD model
    lsdMdlPerm1 = fitglm(trainXlsd1,permTrainYLSD1,'distribution', ...
        'binomial','weights',permWeightLSD1);
    % Test permuted LSD model
    scores = predict(lsdMdlPerm1,testXlsd1);
    % Get permuted LSD model AUCs
    [~,~,~,lsdA_glm80_1hr_perm(k)] = perfcurve(testYlsd1,scores,1);
    % % Break down accuracy for each rat
    % for jj = 1:numel(lsd_ids1)
    %     if ~isempty(subPostTestLSD1{jj}) && ~isempty(subPreTestLSD1{jj})
    %     scores = predict(lsdMdl1,[subPostTestLSD1{jj};subPreTestLSD1{jj}]);
    %     [~,~,~,lsdA_glm80_1hr_ind(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestLSD1{jj}),1);...
    %         zeros(height(subPreTestLSD1{jj}),1)],scores, 1);
    %     scores = predict(lsdMdlPerm1,[subPostTestLSD1{jj}; ...
    %         subPreTestLSD1{jj}]);
    %     [~,~,~,lsdA_glm80_1hr_ind_perm(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestLSD1{jj}),1);...
    %         zeros(height(subPreTestLSD1{jj}),1)],scores, 1);
    %     else
    %         lsdA_glm80_1hr_ind(jj,k) = NaN;
    %         lsdA_glm80_1hr_ind_perm(jj,k) = NaN;
    %     end
    % end

    % Single feature models
    for f = 1:216
        % Build real SAL model
        salMdlSingle1 = fitglm(trainXsal1(:,f),trainYsal1,'distribution', ...
            'binomial','weights',weightSAL1);
        % Extrat sign of beta coefficient from real SAL model
        postSignSAL1(f,k) = sign(table2array(salMdlSingle1.Coefficients(2,1)));
        % Test real SAL model
        scores = predict(salMdlSingle1,testXsal1(:,f));
        % Get real SAL model AUCs
        [~,~,~,salA_glm80_1hr_single(f,k)] = perfcurve(testYsal1,scores,1);
        % Build permuted SAL model
        salMdlSinglePerm1 = fitglm(trainXsal1(:,f),permTrainYSAL1, ...
            'distribution','binomial','weights',permWeightSAL1);
        % Extract sign of permuted beta coefficient from permuted SAL model
        postSignPermSAL1(f,k) = sign(table2array( ...
            salMdlSinglePerm1.Coefficients(2,1)));
        % Test permuted SAL single feature model
        scores = predict(salMdlSinglePerm1,testXsal1(:,f));
        % Get permuted SAL single feature model AUCs
        [~,~,~,salA_glm80_1hr_single_perm(f,k)] = perfcurve(testYsal1, ...
            scores,1);

        % Build real LSD single feature model
        lsdMdlSingle1 = fitglm(trainXlsd1(:,f),trainYlsd1,'distribution', ...
            'binomial','weights',weightLSD1);
        % Extract sign of beta coefficient from real LSD single feature
        % model
        postSignLSD1(f,k) = sign(table2array( ...
            lsdMdlSingle1.Coefficients(2,1)));
        % Test real LSD single feature model
        scores = predict(lsdMdlSingle1,testXlsd1(:,f));
        % Get real LSD single feature model AUCs
        [~,~,~,lsdA_glm80_2hr_single(f,k)] = perfcurve(testYlsd1,scores,1);
        % Build permuted LSD single feature model
        lsdMdlSinglePerm1 = fitglm(trainXlsd1(:,f),permTrainYLSD1, ...
            'distribution','binomial','weights',permWeightLSD1);
        % Extract sign of beta coefficient from permuted LSD single feature
        % model
        postSignPermLSD1(f,k) = sign(table2array( ...
            lsdMdlSinglePerm1.Coefficients(2,1)));
        % Test permuted LSD single feature model
        scores = predict(lsdMdlSinglePerm1,testXlsd1(:,f));
        % Get permuted LSD single feature model AUCs
        [~,~,~,lsdA_glm80_1hr_single_perm(f,k)] = perfcurve(testYlsd1, ...
            scores,1);
    end
    %% Pre vs. post - 2 hour stim (Extended Data Figure 4B)
    % Prepare SAL data
    trainXsal2 = cat(1,subPostTrainSAL2{:},subPreTrainSAL2{:});
    trainYsal2 = [ones(height(cat(1,subPostTrainSAL2{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL2{:})),1)];
    testXsal2 = cat(1,subPostTestSAL2{:},subPreTestSAL2{:});
    testYsal2 = [ones(height(cat(1,subPostTestSAL2{:})),1); ...
        zeros(height(cat(1,subPreTestSAL2{:})),1)];
    % Get SAL weights
    weightSAL2 = cat(1,postSALW2{:},preSALW2{:});
    % Build SAL model
    salMdl2 = fitglm(trainXsal2,trainYsal2,'distribution','binomial', ...
        'weights',weightSAL2);
    % Test SAL model
    scores = predict(salMdl2,testXsal2);
    % Get real SAL model AUCs
    [~,~,~,salA_glm80_2hr(k)] = perfcurve(testYsal2,scores,1);
    % Permuted SAL data
    permInd = randperm(numel(trainYsal2),numel(trainYsal2));
    permTrainYSAL2 = trainYsal2(permInd);
    permWeightSAL2 = weightSAL2(permInd);
    % Build permuted SAL model
    salMdlPerm2 = fitglm(trainXsal2,permTrainYSAL2,'distribution', ...
        'binomial','weights',permWeightSAL2);
    % Test permuted SAL model
    scores = predict(salMdlPerm2,testXsal2);
    % Get permuted SAL model AUCs
    [~,~,~,salA_glm80_2hr_perm(k)] = perfcurve(testYsal2,scores,1);
    % % Break down accuracy for each rat
    % for jj = 1:numel(sal_ids2)
    %     if ~isempty(subPostTestSAL2{jj}) && ~isempty(subPreTestSAL2{jj})
    %         scores = predict(salMdl2,[subPostTestSAL2{jj}; ...
    %             subPreTestSAL2{jj}]);
    %         [~,~,~,salA_glm80_2hr_ind(jj,k)]  = perfcurve([ones(height( ...
    %             subPostTestSAL2{jj}),1);...
    %             zeros(height(subPreTestSAL2{jj}),1)],scores, 1);
    %         scores = predict(salMdlPerm2,[subPostTestSAL2{jj}; ...
    %             subPreTestSAL2{jj}]);
    %         [~,~,~,salA_glm80_2hr_ind_perm(jj,k)]  = perfcurve([ones( ...
    %             height(subPostTestSAL2{jj}),1); ...
    %             zeros(height(subPreTestSAL2{jj}),1)],scores, 1);
    %     else
    %         salA_glm80_2hr_ind(jj,k) = NaN;
    %         salA_glm80_2hr_ind_perm(jj,k) = NaN;
    %     end
    % end

    % Prepare real LSD data
    trainXlsd2 = cat(1,subPostTrainLSD2{:},subPreTrainLSD2{:});
    trainYlsd2 = [ones(height(cat(1,subPostTrainLSD2{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD2{:})),1)];
    testXlsd2 = cat(1,subPostTestLSD2{:},subPreTestLSD2{:});
    testYlsd2 = [ones(height(cat(1,subPostTestLSD2{:})),1); ...
        zeros(height(cat(1,subPreTestLSD2{:})),1)];
    % Get real LSD weights
    weightLSD2 = cat(1,postLSDW2{:},preLSDW2{:});
    % Build real LSD model
    lsdMdl2 = fitglm(trainXlsd2,trainYlsd2,'distribution','binomial', ...
        'weights',weightLSD2);
    % Test real LSD model
    scores = predict(lsdMdl2,testXlsd2);
    % Get real LSD model AUCs
    [~,~,~,lsdA_glm80_2hr(k)] = perfcurve(testYlsd2,scores,1);
    % Permute LSD data
    permInd = randperm(numel(trainYlsd2),numel(trainYlsd2));
    permTrainYLSD2 = trainYlsd2(permInd);
    permWeightLSD2 = weightLSD2(permInd);
    % Build permuted LSD model
    lsdMdlPerm2 = fitglm(trainXlsd2,permTrainYLSD2,'distribution', ...
        'binomial','weights',permWeightLSD2);
    % Test permuted LSD model
    scores = predict(lsdMdlPerm2,testXlsd2);
    % Get permuted LSD model AUCs
    [~,~,~,lsdA_glm80_2hr_perm(k)] = perfcurve(testYlsd2,scores,1);
    % % Break down accuracy for each rat
    % for jj = 1:numel(lsd_ids2)
    %     if ~isempty(subPostTestLSD2{jj}) && ~isempty(subPreTestLSD2{jj})
    %     scores = predict(lsdMdl2,[subPostTestLSD2{jj};subPreTestLSD2{jj}]);
    %     [~,~,~,lsdA_glm80_2hr_ind(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestLSD2{jj}),1);...
    %         zeros(height(subPreTestLSD2{jj}),1)],scores, 1);
    %     scores = predict(lsdMdlPerm2,[subPostTestLSD2{jj}; ...
    %         subPreTestLSD2{jj}]);
    %     [~,~,~,lsdA_glm80_2hr_ind_perm(jj,k)]  = perfcurve([ones(height( ...
    %         subPostTestLSD2{jj}),1);...
    %         zeros(height(subPreTestLSD2{jj}),1)],scores, 1);
    %     else
    %         lsdA_glm80_2hr_ind(jj,k) = NaN;
    %         lsdA_glm80_2hr_ind_perm(jj,k) = NaN;
    %     end
    % end

    % Single feature models
    for f = 1:216
        % Build real SAL model
        salMdlSingle2 = fitglm(trainXsal2(:,f),trainYsal2,'distribution', ...
            'binomial','weights',weightSAL2);
        % Extrat sign of beta coefficient from real SAL model
        postSignSAL2(f,k) = sign(table2array(salMdlSingle2.Coefficients(2,1)));
        % Test real SAL model
        scores = predict(salMdlSingle2,testXsal2(:,f));
        % Get real SAL model AUCs
        [~,~,~,salA_glm80_2hr_single(f,k)] = perfcurve(testYsal2,scores,1);
        % Build permuted SAL model
        salMdlSinglePerm2 = fitglm(trainXsal2(:,f),permTrainYSAL2, ...
            'distribution','binomial','weights',permWeightSAL2);
        % Extract sign of permuted beta coefficient from permuted SAL model
        postSignPermSAL2(f,k) = sign(table2array( ...
            salMdlSinglePerm2.Coefficients(2,1)));
        % Test permuted SAL single feature model
        scores = predict(salMdlSinglePerm2,testXsal2(:,f));
        % Get permuted SAL single feature model AUCs
        [~,~,~,salA_glm80_2hr_single_perm(f,k)] = perfcurve(testYsal2, ...
            scores,1);

        % Build real LSD single feature model
        lsdMdlSingle2 = fitglm(trainXlsd2(:,f),trainYlsd2,'distribution', ...
            'binomial','weights',weightLSD2);
        % Extract sign of beta coefficient from real LSD single feature
        % model
        postSignLSD2(f,k) = sign(table2array( ...
            lsdMdlSingle2.Coefficients(2,1)));
        % Test real LSD single feature model
        scores = predict(lsdMdlSingle2,testXlsd2(:,f));
        % Get real LSD single feature model AUCs
        [~,~,~,lsdA_glm80_2hr_single(f,k)] = perfcurve(testYlsd2,scores,1);
        % Build permuted LSD single feature model
        lsdMdlSinglePerm2 = fitglm(trainXlsd2(:,f),permTrainYLSD2, ...
            'distribution','binomial','weights',permWeightLSD2);
        % Extract sign of beta coefficient from permuted LSD single feature
        % model
        postSignPermLSD2(f,k) = sign(table2array( ...
            lsdMdlSinglePerm2.Coefficients(2,1)));
        % Test permuted LSD single feature model
        scores = predict(lsdMdlSinglePerm2,testXlsd2(:,f));
        % Get permuted LSD single feature model AUCs
        [~,~,~,lsdA_glm80_2hr_single_perm(f,k)] = perfcurve(testYlsd2, ...
            scores,1);
    end
    %% Pre vs. stim - combined cohorts (Figure 4A)
    % Prepare SAL data
    trainXstimSALcomb = cat(1,subStimTrainSAL2{:},subPreTrainSAL2{:}, ...
        subStimTrainSAL1{:},subPreTrainSAL1{:});
    trainYstimSALcomb = [ones(height(cat(1,subStimTrainSAL2{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL2{:})),1); ...
        ones(height(cat(1,subStimTrainSAL1{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL1{:})),1)];
    testXstimSALcomb = cat(1,subStimTestSAL2{:},subPreTestSAL2{:}, ...
        subStimTestSAL1{:},subPreTestSAL1{:});
    testYstimSALcomb = [ones(height(cat(1,subStimTestSAL2{:})),1); ...
        zeros(height(cat(1,subPreTestSAL2{:})),1); ...
        ones(height(cat(1,subStimTestSAL1{:})),1); ...
        zeros(height(cat(1,subPreTestSAL1{:})),1)];
    % Get SAL weights
    weightStimSALcomb = cat(1,stimSALW2{:},preSALW2{:},stimSALW1{:}, ...
        preSALW1{:});
    % Build SAL model
    salMdlStimComb = fitglm(trainXstimSALcomb,trainYstimSALcomb, ...
        'distribution','binomial','weights',weightStimSALcomb);
    % Test SAL model
    scores = predict(salMdlStimComb,testXstimSALcomb);
    % Get SAL model AUCs
    [~,~,~,salA_glm80_stim_comb(k)] = perfcurve(testYstimSALcomb,scores,1);

    % Prepare LSD data
    trainXstimLSDcomb = cat(1,subStimTrainLSD2{:},subPreTrainLSD2{:}, ...
        subStimTrainLSD1{:},subPreTrainLSD1{:});
    trainYstimLSDcomb = [ones(height(cat(1,subStimTrainLSD2{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD2{:})),1); ...
        ones(height(cat(1,subStimTrainLSD1{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD1{:})),1)];
    testXstimLSDcomb = cat(1,subStimTestLSD2{:},subPreTestLSD2{:}, ...
        subStimTestLSD1{:},subPreTestLSD1{:});
    testYstimLSDcomb = [ones(height(cat(1,subStimTestLSD2{:})),1); ...
        zeros(height(cat(1,subPreTestLSD2{:})),1); ...
        ones(height(cat(1,subStimTestLSD1{:})),1); ...
        zeros(height(cat(1,subPreTestLSD1{:})),1)];
    % Get LSD weights
    weightStimLSDcomb = cat(1,stimLSDW2{:},preLSDW2{:},stimLSDW1{:}, ...
        preLSDW1{:});
    % Build LSD model
    lsdMdlStimComb = fitglm(trainXstimLSDcomb,trainYstimLSDcomb, ...
        'distribution','binomial','weights',weightStimLSDcomb);
    % Test LSD model
    scores = predict(lsdMdlStimComb,testXstimLSDcomb);
    % Get LSD model AUCs
    [~,~,~,lsdA_glm80_stim_comb(k)] = perfcurve(testYstimLSDcomb,scores,1);
     %% Pre vs. stim - 1 hr stim
    % Prepare SAL data
    trainXstimSAL1 = cat(1,subStimTrainSAL1{:},subPreTrainSAL1{:});
    trainYstimSAL1 = [ones(height(cat(1,subStimTrainSAL1{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL1{:})),1)];
    testXstimSAL1 = cat(1,subStimTestSAL1{:},subPreTestSAL1{:});
    testYstimSAL1 = [ones(height(cat(1,subStimTestSAL1{:})),1); ...
        zeros(height(cat(1,subPreTestSAL1{:})),1)];
    % Get SAL weights
    weightStimSAL1 = cat(1,stimSALW1{:},preSALW1{:});
    % Build SAL model
    salMdlStim1 = fitglm(trainXstimSAL1,trainYstimSAL1,'distribution', ...
        'binomial','weights',weightStimSAL1);
    % Test SAL model
    scores = predict(salMdlStim1,testXstimSAL1);
    % Get SAL model AUCs
    [~,~,~,salA_glm80_1hr_stim(k)] = perfcurve(testYstimSAL1,scores,1);

    % Prepare LSD data
    trainXstimLSD1 = cat(1,subStimTrainLSD1{:},subPreTrainLSD1{:});
    trainYstimLSD1 = [ones(height(cat(1,subStimTrainLSD1{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD1{:})),1)];
    testXstimLSD1 = cat(1,subStimTestLSD1{:},subPreTestLSD1{:});
    testYstimLSD1 = [ones(height(cat(1,subStimTestLSD1{:})),1); ...
        zeros(height(cat(1,subPreTestLSD1{:})),1)];
    % Get LSD weights
    weightStimLSD1 = cat(1,stimLSDW1{:},preLSDW1{:});
    % Build LSD model
    lsdMdlStim1 = fitglm(trainXstimLSD1,trainYstimLSD1,'distribution', ...
        'binomial','weights',weightStimLSD1);
    % Test LSD model
    scores = predict(lsdMdlStim1,testXstimLSD1);
    % Get LSD AUCs
    [~,~,~,lsdA_glm80_1hr_stim(k)] = perfcurve(testYstimLSD1,scores,1);
    %% Pre vs. stim - 2 hr stim
    % Prepare SAL data
    trainXstimSAL2 = cat(1,subStimTrainSAL2{:},subPreTrainSAL2{:});
    trainYstimSAL2 = [ones(height(cat(1,subStimTrainSAL2{:})),1); ...
        zeros(height(cat(1,subPreTrainSAL2{:})),1)];
    testXstimSAL2 = cat(1,subStimTestSAL2{:},subPreTestSAL2{:});
    testYstimSAL2 = [ones(height(cat(1,subStimTestSAL2{:})),1); ...
        zeros(height(cat(1,subPreTestSAL2{:})),1)];
    % Get SAL weights
    weightStimSAL2 = cat(1,stimSALW2{:},preSALW2{:});
    % Build SAL model
    salMdlStim2 = fitglm(trainXstimSAL2,trainYstimSAL2,'distribution', ...
        'binomial','weights',weightStimSAL2);
    % Test SAL model
    scores = predict(salMdlStim2,testXstimSAL2);
    % Get SAL model AUCs
    [~,~,~,salA_glm80_2hr_stim(k)] = perfcurve(testYstimSAL2,scores,1);

    % Prepare LSD data
    trainXstimLSD2 = cat(1,subStimTrainLSD2{:},subPreTrainLSD2{:});
    trainYstimLSD2 = [ones(height(cat(1,subStimTrainLSD2{:})),1); ...
        zeros(height(cat(1,subPreTrainLSD2{:})),1)];
    testXstimLSD2 = cat(1,subStimTestLSD2{:},subPreTestLSD2{:});
    testYstimLSD2 = [ones(height(cat(1,subStimTestLSD2{:})),1); ...
        zeros(height(cat(1,subPreTestLSD2{:})),1)];
    % Get LSD weights
    weightStimLSD2 = cat(1,stimLSDW2{:},preLSDW2{:});
    % Build LSD model
    lsdMdlStim2 = fitglm(trainXstimLSD2,trainYstimLSD2,'distribution', ...
        'binomial','weights',weightStimLSD2);
    % Test LSD model
    scores = predict(lsdMdlStim2,testXstimLSD2);
    % Get LSD AUCs
    [~,~,~,lsdA_glm80_2hr_stim(k)] = perfcurve(testYstimLSD2,scores,1);
    %% Pre vs. stim single feature
    for f = 1:216
    %     %% Combined cohorts
    %     % Build SAL single feature model
    %     salMdlSingleCombStim = fitglm(trainXstimSALcomb(:,f), ...
    %         trainYstimSALcomb,'distribution', ...
    %         'binomial','weights',weightStimSALcomb);
    %     % Extract sign of beta coeffiecent from SAL single feature
    %     % model
    %     postSignSALcombStim(f,k) = sign(table2array( ...
    %         salMdlSingleCombStim.Coefficients(2,1)));
    %     % Test SAL single feature model
    %     scores = predict(salMdlSingleCombStim, ...
    %         testXstimSALcomb(:,f));
    %     % Get SAL single feature model AUCs
    %     [~,~,~,salA_glm80_comb_single_stim(f,k)] = perfcurve( ...
    %         testYstimSALcomb,scores,1);
    % 
        % Build LSD single feature model
        lsdMdlSingleCombStim = fitglm(trainXstimLSDcomb(:,f), ...
            trainYstimLSDcomb,'distribution',...
            'binomial','weights',weightStimLSDcomb);
        % Extract sign of beta coeffiecent from LSD single feature
        % model
        postSignLSDcombStim(f,k) = sign(table2array( ...
            lsdMdlSingleCombStim.Coefficients(2,1)));
        % Test LSD single feature model
        scores = predict(lsdMdlSingleCombStim, ...
            testXstimLSDcomb(:,f));
        % Get LSD single feature model AUCs
        [~,~,~,lsdA_glm80_comb_single_stim(f,k)] = perfcurve( ...
            testYstimLSDcomb,scores,1);
    %     %% 1 hr stim
    %     % Build SAL single feature model
    %     salMdlSingleStim1{f} = fitglm(trainXstimSAL1(:,f), ...
    %         trainYstimSAL1,'distribution','binomial','weights', ...
    %         weightStimSAL1);
    %     % Extract sign of beta coeffiecent from SAL single feature
    %     % model
    %     postSignSALstim1(f,k) = sign(table2array( ...
    %         salMdlSingleStim1{f}.Coefficients(2,1)));
    %     % Test SAL single feature model
    %     scores = predict(salMdlSingleStim1{f},testXstimSAL1(:,f));
    %     % Get SAL single feature model AUCs
    %     [~,~,~,salA_glm80_1hr_single_stim(f,k)] = perfcurve( ...
    %         testYstimSAL1,scores,1);
    % 
    %     % Build LSD single feature model
    %     lsdMdlSingleStim1{f} = fitglm(trainXstimLSD1(:,f), ...
    %         trainYstimLSD1,'distribution','binomial','weights', ...
    %         weightStimLSD1);
    %     % Extract sign of beta coeffiecent from LSD single feature
    %     % model
    %     postSignLSDstim1(f,k) = sign(table2array( ...
    %         lsdMdlSingleStim1{f}.Coefficients(2,1)));
    %     % Test LSD single feature model
    %     scores = predict(lsdMdlSingleStim1{f},testXstimLSD1(:,f));
    %     % Get LSD single feature model AUCs
    %     [~,~,~,lsdA_glm80_1hr_single_stim(f,k)] = perfcurve(testYstimLSD1, ...
    %         scores,1);
    %     %% 2 hr stim
    %     % Build SAL single feature model
    %     salMdlSingleStim2{f} = fitglm(trainXstimSAL2(:,f), ...
    %         trainYstimSAL2,'distribution','binomial','weights', ...
    %         weightStimSAL2);
    %     % Extract sign of beta coeffiecent from SAL single feature
    %     % model
    %     postSignSALstim2(f,k) = sign(table2array( ...
    %         salMdlSingleStim2{f}.Coefficients(2,1)));
    %     % Test SAL single feature model
    %     scores = predict(salMdlSingleStim2{f},testXstimSAL2(:,f));
    %     % Get SAL single feature model AUCs
    %     [~,~,~,salA_glm80_2hr_single_stim(f,k)] = perfcurve( ...
    %         testYstimSAL2,scores,1);
    % 
    %     % Build LSD single feature model
    %     lsdMdlSingleStim2{f} = fitglm(trainXstimLSD2(:,f), ...
    %         trainYstimLSD2,'distribution','binomial','weights', ...
    %         weightStimLSD2);
    %     % Extract sign of beta coeffiecent from LSD single feature
    %     % model
    %     postSignLSDstim2(f,k) = sign(table2array( ...
    %         lsdMdlSingleStim2{f}.Coefficients(2,1)));
    %     % Test LSD single feature model
    %     scores = predict(lsdMdlSingleStim2{f},testXstimLSD2(:,f));
    %     % Get LSD single feature model AUCs
    %     [~,~,~,lsdA_glm80_2hr_single_stim(f,k)] = perfcurve(testYstimLSD2, ...
    %         scores,1);
    end
    %% Pre vs. stim applied to post
    % Combined
    % Test SAL stim model on post data
    scores = predict(salMdlStimComb,testXsalComb);
    % Get SAL stim model → post data AUCs
    [~,~,~,salA_glm80_comb_stim_post(k)] = perfcurve(testYsalComb,scores, ...
        1);
    % Test LSD stim model on post data
    scores = predict(lsdMdlStimComb,testXlsdComb);
    % Get LSD stim model → post data AUCs
    [~,~,~,lsdA_glm80_comb_stim_post(k)] = perfcurve(testYlsdComb,scores, ...
        1);
    
    % 1 hr stim
    % Test SAL stim model on post data
    scores = predict(salMdlStim1,testXsal1);
    % Get SAL stim model → post data AUCs
    [~,~,~,salA_glm80_1hr_stim_post(k)] = perfcurve(testYsal1,scores,1);
    % Test LSD stim model on post data
    scores = predict(lsdMdlStim1,testXlsd1);
    % Get LSD stim model → post data AUCs
    [~,~,~,lsdA_glm80_1hr_stim_post(k)] = perfcurve(testYlsd1,scores,1);

    % 2 hr stim
    % Test SAL stim model on post data
    scores = predict(salMdlStim2,testXsal2);
    % Get SAL stim model → post data AUCs
    [~,~,~,salA_glm80_2hr_stim_post(k)] = perfcurve(testYsal2,scores,1);
    % Test LSD stim model on post data
    scores = predict(lsdMdlStim2,testXlsd2);
    % Get LSD stim model → post data AUCs
    [~,~,~,lsdA_glm80_2hr_stim_post(k)] = perfcurve(testYlsd2,scores,1);
    %% Cross apply models (LSD↔SAL) - combined cohorts (Figure 3D)
    % Test LSD model on SAL data (LSD→SAL)
    scores = predict(lsdMdlComb,testXsalComb);
    % Get LSD→SAL AUCs
    [~,~,~,salA_glm80_comb_lsdTrain(k)] = perfcurve(testYsalComb,scores,1);  
    % Test SAL model on LSD data (SAL→LSD)
    scores = predict(salMdlComb,testXlsdComb);
    % Get SAL→LSD AUCs
    [~,~,~,lsdA_glm80_comb_salTrain(k)] = perfcurve(testYlsdComb,scores,1);    
    %% Cross apply models (LSD↔SAL) - 1 hr (Extended Data Figure 4E)
    % Test LSD model on SAL data (LSD→SAL)
    scores = predict(lsdMdl1,testXsal1);
    % Get LSD→SAL AUCs
    [~,~,~,salA_glm80_1hr_lsdTrain(k)] = perfcurve(testYsal1,scores,1);    
    % Test SAL model on LSD data (SAL→LSD)
    scores = predict(salMdl1,testXlsd1);
    % Get SAL→LSD AUCs
    [~,~,~,lsdA_glm80_1hr_salTrain(k)] = perfcurve(testYlsd1,scores,1);
    %% Cross apply models (LSD↔SAL) - 2 hr (Extended Data Figure 4D)
    % Test LSD model on SAL data (LSD→SAL)
    scores = predict(lsdMdl2,testXsal2);
    % Get LSD→SAL AUCs
    [~,~,~,salA_glm80_2hr_lsdTrain(k)] = perfcurve(testYsal2,scores,1);    
    % Test SAL model on LSD data (SAL→LSD)
    scores = predict(salMdl2,testXlsd2);
    % Get SAL→LSD AUCs
    [~,~,~,lsdA_glm80_2hr_salTrain(k)] = perfcurve(testYlsd2,scores,1);
    %% Models applied to 5 min post chunks - combined cohorts
    % Grab unused postData
    subPostLeftSAL = cell(numel(sal_ids_all),1);
    [binSAL,preBinSAL,postBinSAL] = deal(cell(numel(sal_ids_all),12));
    subPostLeftLSD = cell(numel(lsd_ids_all),1);
    [binLSD,preBinLSD,postBinLSD] = deal(cell(numel(lsd_ids_all),12));

    % Combine data
    postIndSALall = [postIndSAL2;postIndSAL1];
    postIndLSDall = [postIndLSD2;postIndLSD1];
    postSALall = [postSAL2,postSAL1];
    postLSDall = [postLSD2,postLSD1];
    subPreSALall = [subPreSAL2;subPreSAL1];
    subPreLSDall = [subPreLSD2;subPreLSD1];
    for ii = 1:12
        start = (ii-1)*300;
        stop = ii*300;
        % SAL
        for jj = 1:numel(sal_ids_all)
            subPostLeftSAL{jj} = postSALall{jj}(~ismember(1: ...
                height(postSALall{jj}),postIndSALall{jj}),:);

            % Find data in this chunk
            binSAL{jj,ii} = subPostLeftSAL{jj}.normData( ...
                subPostLeftSAL{jj}.stimTime>=start & ...
                subPostLeftSAL{jj}.stimTime<=stop,:);

            % Balance each rat with pre data
            if ~isempty(binSAL{jj,ii}) && ~isempty(subPreSALall{jj})
                n = min([height(binSAL{jj,ii}) height(subPreSALall{jj})]);
                preBinSAL{jj,ii} = subPreSALall{jj}(randperm(height( ...
                    subPreSALall{jj}),n),:);
                postBinSAL{jj,ii} = binSAL{jj,ii}(randperm(height( ...
                    binSAL{jj,ii}),n),:);

                % Test on individual rats
                testXsal = [preBinSAL{jj,ii}; postBinSAL{jj,ii}];
                testYsal = [zeros(n,1); ones(n,1)];

                % GLM
                % Pre vs. post
                scores = predict(salMdlComb,testXsal);
                [~,~,~,salA_glm80_comb_postBinInd(jj,ii,k)] = ...
                    perfcurve(testYsal,scores,1);
            else
                salA_glm80_comb_postBinInd(jj,ii,k) = NaN;
            end
        end
        % LSD
        for jj = 1:numel(lsd_ids_all)
            subPostLeftLSD{jj} = postLSDall{jj}(~ismember(1: ...
                height(postLSDall{jj}),postIndLSDall{jj}),:);

            % Find data in this chunk
            binLSD{jj,ii} = subPostLeftLSD{jj}.normData( ...
                subPostLeftLSD{jj}.stimTime>=start & ...
                subPostLeftLSD{jj}.stimTime<=stop,:);
            % Balance each rat with pre data
            if ~isempty(binLSD{jj,ii}) && ~isempty(subPreLSDall{jj})
                n = min([height(binLSD{jj,ii}) height(subPreLSDall{jj})]);
                preBinLSD{jj,ii} = subPreLSDall{jj}(randperm(height( ...
                    subPreLSDall{jj}),n),:);
                postBinLSD{jj,ii} = binLSD{jj,ii}(randperm(height( ...
                    binLSD{jj,ii}),n),:);

                % Test on individual rats
                testXlsd = [preBinLSD{jj,ii}; postBinLSD{jj,ii}];
                testYlsd = [zeros(n,1); ones(n,1)];

                % Pre vs. post
                scores = predict(lsdMdlComb,testXlsd);
                [~,~,~,lsdA_glm80_comb_postBinInd(jj,ii,k)] = ...
                perfcurve(testYlsd,scores,1);
            else
                lsdA_glm80_comb_postBinInd(jj,ii,k) = NaN;
            end
        end
    end
    %% Test models on 1 hr washout days
    % Grab an equal number of wash data as preTest data
    [washTestXsal1,washTestYsal1] = deal(cell(numel(sal_ids1), ...
        numel(times1)));
    [washTestXlsd1,washTestYlsd1] = deal(cell(numel(lsd_ids1), ...
        numel(times1)));
    for ii = 1:numel(times1)
        % SAL
        for jj = 1:numel(sal_ids1)
            if ~isempty(washSAL1{jj,ii})
                washTestXsal1{jj,ii} = [subPreTestSAL1{jj};
                    washSAL1{jj,ii}.normData(randperm( ...
                    height(washSAL1{jj,ii}), min([height(washSAL1{jj,ii}) ...
                    height(subPostTestSAL1{jj})])),:)];
                washTestYsal1{jj,ii} = [zeros(height(subPreTestSAL1{jj}),1); ...
                    ones(height(subPreTestSAL1{jj}),1)];
            end
        end
        % LSD
        for jj = 1:numel(lsd_ids1)
            if ~isempty(washLSD1{jj,ii})
                washTestXlsd1{jj,ii} = [subPreTestLSD1{jj};
                    washLSD1{jj,ii}.normData(randperm(...
                    height(washLSD1{jj,ii}), min([height(washLSD1{jj,ii}) ...
                    height(subPostTestLSD1{jj})])),:)];
                washTestYlsd1{jj,ii} = [zeros(height(subPreTestLSD1{jj}),1); ...
                    ones(height(subPreTestLSD1{jj}),1)];
            end
        end
        % Pre vs. post
        % scores = predict(salMdl2,cat(1,washTestXsal1{:,ii}));
        % [~,~,~,salA_glm80_wash_21(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal1{:,ii}),scores,1);
        % 
        % scores = predict(lsdMdl2,cat(1,washTestXlsd1{:,ii}));
        % [~,~,~,lsdA_glm80_wash_21(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd1{:,ii}),scores,1);
        
        % 1 hr pre vs. post model
        scores = predict(salMdl1,cat(1,washTestXsal1{:,ii}));
        [~,~,~,salA_glm80_1hr_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii}),scores,1);

        scores = predict(lsdMdl1,cat(1,washTestXlsd1{:,ii}));
        [~,~,~,lsdA_glm80_1hr_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii}),scores,1);
        
        % Combined pre vs. post model
        scores = predict(salMdlComb,cat(1,washTestXsal1{:,ii}));
        [~,~,~,salA_glm80_comb1_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii}),scores,1);

        scores = predict(lsdMdlComb,cat(1,washTestXlsd1{:,ii}));
        [~,~,~,lsdA_glm80_comb1_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii}),scores,1);

        scores = predict(salMdlPermComb,cat(1,washTestXsal1{:,ii}));
        [~,~,~,salA_glm80_comb1_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii}),scores,1);

        scores = predict(lsdMdlPermComb,cat(1,washTestXlsd1{:,ii}));
        [~,~,~,lsdA_glm80_comb1_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii}),scores,1);

        % scores = predict(salMdlComb,cat(1,washTestXlsd1{:,ii}));
        % [~,~,~,lsdA_glm80_comb1_wash_salTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd1{:,ii}),scores,1);
        % 
        % scores = predict(lsdMdlComb,cat(1,washTestXsal1{:,ii}));
        % [~,~,~,salA_glm80_comb1_wash_lsdTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal1{:,ii}),scores,1);

        % Pre vs. stim
        scores = predict(salMdlStimComb,cat(1,washTestXsal1{:,ii}));
        [~,~,~,salA_glm80_comb1_wash_stim(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii}),scores,1);

        scores = predict(lsdMdlStimComb,cat(1,washTestXlsd1{:,ii}));
        [~,~,~,lsdA_glm80_comb1_wash_stim(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii}),scores,1);

        % scores = predict(salMdlStimPermComb{k},cat(1,washTestXsal1{:,ii}));
        % [~,~,~,salA_glm80_comb1_wash_perm_stim(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal1{:,ii}),scores,1);
        % 
        % scores = predict(lsdMdlStimPermComb{k},cat(1,washTestXlsd1{:,ii}));
        % [~,~,~,lsdA_glm80_comb1_wash_perm_stim(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd1{:,ii}),scores,1);

         % Single feature
        lsdX = cat(1,washTestXlsd1{:,ii});
        lsdY = cat(1,washTestYlsd1{:,ii});
        salX = cat(1,washTestXsal1{:,ii});
        salY = cat(1,washTestYsal1{:,ii});
        for f = 1:216
            % Pre vs. post
            scores = predict(salMdlSingle1,salX(:,f));
            [~,~,~,salA_glm80_1hr_wash_single(ii,f,k)] = perfcurve(salY, ...
                scores,1);
            scores = predict(lsdMdlSingle1,lsdX(:,f));
            [~,~,~,lsdA_glm80_1hr_wash_single(ii,f,k)] = perfcurve(lsdY, ...
                scores,1);
        end
    end
    %% Test models on 2 hr washout days
    % Grab an equal number of wash data as preTest data
    [washTestXsal2,washTestYsal2] = deal(cell(numel(sal_ids2),numel(times2)));
    [washTestXlsd2,washTestYlsd2] = deal(cell(numel(lsd_ids2),numel(times2)));
    for ii = 1:numel(times2)
        % SAL
        for jj = 1:numel(sal_ids2)
            if ~isempty(washSAL2{jj,ii})
                washTestXsal2{jj,ii} = [subPreTestSAL2{jj};
                    washSAL2{jj,ii}.normData(randperm(...
                    height(washSAL2{jj,ii}), min([height(washSAL2{jj,ii}) ...
                    height(subPostTestSAL2{jj})])),:)];
                washTestYsal2{jj,ii} = [zeros(height(subPreTestSAL2{jj}),1); ...
                    ones(height(subPreTestSAL2{jj}),1)];

                % % Ind
                % scores = predict(salMdl2,washTestXsal2{jj,ii});
                % [~,~,~,salA_glm80_2hr_washInd(jj,ii,k)] = perfcurve( ...
                %     washTestYsal2{jj,ii},scores,1);
                % scores = predict(salMdlPerm2,washTestXsal2{jj,ii});
                % [~,~,~,salA_glm80_2hr_washInd_perm(jj,ii,k)] = perfcurve( ...
                %     washTestYsal2{jj,ii},scores,1);
            end
        end
        % LSD
        for jj = 1:numel(lsd_ids2)
            if ~isempty(washLSD2{jj,ii})
                washTestXlsd2{jj,ii} = [subPreTestLSD2{jj};
                    washLSD2{jj,ii}.normData(randperm(...
                    height(washLSD2{jj,ii}), min([height(washLSD2{jj,ii}) ...
                    height(subPostTestLSD2{jj})])),:)];
                washTestYlsd2{jj,ii} = [zeros(height(subPreTestLSD2{jj}),1); ...
                    ones(height(subPreTestLSD2{jj}),1)];

                % % Ind
                % scores = predict(lsdMdl2,washTestXlsd2{jj,ii});
                % [~,~,~,lsdA_glm80_2hr_washInd(jj,ii,k)] = perfcurve( ...
                %     washTestYlsd2{jj,ii},scores,1);
                % scores = predict(lsdMdlPerm2,washTestXlsd2{jj,ii});
                % [~,~,~,lsdA_glm80_2hr_washInd_perm(jj,ii,k)] = perfcurve( ...
                %     washTestYlsd2{jj,ii},scores,1);
            end
        end
        % % Pre vs. post: combined
        % scores = predict(salMdlComb,cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_comb2_wash(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % scores = predict(salMdlPermComb,cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_comb2_wash_perm(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % 
        % scores = predict(lsdMdlComb,cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_comb2_wash(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);
        % scores = predict(lsdMdlPermComb,cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_comb2_wash_perm(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);

        % Pre vs. post: 2 hr
        scores = predict(salMdl2,cat(1,washTestXsal2{:,ii}));
        [~,~,~,salA_glm80_2hr_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYsal2{:,ii}),scores,1);
        scores = predict(salMdlPerm2,cat(1,washTestXsal2{:,ii}));
        [~,~,~,salA_glm80_2hr_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestYsal2{:,ii}),scores,1);

        scores = predict(lsdMdl2,cat(1,washTestXlsd2{:,ii}));
        [~,~,~,lsdA_glm80_2hr_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd2{:,ii}),scores,1);
        scores = predict(lsdMdlPerm2,cat(1,washTestXlsd2{:,ii}));
        [~,~,~,lsdA_glm80_2hr_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd2{:,ii}),scores,1);

        % % Pre vs. stim: combined
        % scores = predict(salMdlStimComb,cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_comb2_stim_wash(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % scores = predict(salMdlStimPermComb{k},cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_comb2_stim_wash_perm(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % 
        % scores = predict(lsdMdlStimComb,cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_comb2_stim_wash(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);
        % scores = predict(lsdMdlStimPermComb{k},cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_comb2_stim_wash_perm(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);

        % % Pre vs. stim: 2 hr
        % scores = predict(salMdlStim2,cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_2hr_stim_wash(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % scores = predict(salMdlStimPerm2{k},cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_2hr_stim_wash_perm(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % 
        % scores = predict(lsdMdlStim2,cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_2hr_stim_wash(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);
        % scores = predict(lsdMdlStimPerm2{k},cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_2hr_stim_wash_perm(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);

        % % Cross applied
        % scores = predict(lsdMdl2,cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_2hr_wash_lsdTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % scores = predict(salMdl2,cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_2hr_wash_salTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);
        % scores = predict(lsdMdlStim2,cat(1,washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_2hr_stim_wash_lsdTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal2{:,ii}),scores,1);
        % scores = predict(salMdlStim2,cat(1,washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_2hr_stim_wash_salTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd2{:,ii}),scores,1);

        % Single feature
        lsdX = cat(1,washTestXlsd2{:,ii});
        lsdY = cat(1,washTestYlsd2{:,ii});
        salX = cat(1,washTestXsal2{:,ii});
        salY = cat(1,washTestYsal2{:,ii});
        for f = 1:216
            % Pre vs. post
            scores = predict(salMdlSingle2,salX(:,f));
            [~,~,~,salA_glm80_2hr_wash_single(ii,f,k)] = perfcurve(salY, ...
                scores,1);
            % scores = predict(salMdlSinglePerm{f},cat(1,salX(:,f)));
            % [~,~,~,salA_glm80_wash_single_perm(ii,f,k)] = perfcurve(salY, ...
            %     scores,1);

            scores = predict(lsdMdlSingle2,lsdX(:,f));
            [~,~,~,lsdA_glm80_2hr_wash_single(ii,f,k)] = perfcurve(lsdY, ...
                scores,1);
            % scores = predict(lsdMdlSinglePerm{f},lsdX(:,f));
            % [~,~,~,lsdA_glm80_wash_single_perm(ii,f,k)] = perfcurve(lsdY, ...
            %     scores,1);

            % % Pre vs. stim
            % scores = predict(salMdlSingleStim2{f},salX(:,f));
            % [~,~,~,salA_glm80_2hr_wash_single_stim(ii,f,k)] = ...
            %     perfcurve(salY,scores,1);
            % scores = predict(salMdlSingleStimPerm2{f},salX(:,f));
            % [~,~,~,salA_glm80_wash_single_perm_stim(ii,f,k)] = ...
            %     perfcurve(salY,scores,1);
            % 
            % scores = predict(lsdMdlSingleStim2{f},lsdX(:,f));
            % [~,~,~,lsdA_glm80_wash_single_stim(ii,f,k)] = ...
            %     perfcurve(lsdY,scores,1);
            % scores = predict(lsdMdlSingleStimPerm2{f},lsdX(:,f));
            % [~,~,~,lsdA_glm80_wash_single_perm_stim(ii,f,k)] = ...
            %     perfcurve(lsdY,scores,1);
        end
    end
    %% Test models on combined washout days
    % allWashSAL = [cat(1,washSAL2(:,1:2),washSAL1(:,1:2))];
    % allWashLSD = [cat(1,washLSD2(:,1:2),washLSD1(:,1:2))];
    for ii = 1:numel(times2)
        % % SAL
        % for jj = 1:numel(sal_ids_all)
        %     if ~isempty(allWashSAL{jj,ii})
        %         thisWashTestXsal = [allPreTestSAL{jj}; ...
        %             allWashSAL{jj,ii}.normData(randperm(height( ...
        %             allWashSAL{jj,ii}),min([height( ...
        %             allWashSAL{jj,ii}) height(allPostTestSAL{jj})])),:)];
        % 
        %         thisWashTestYsal = [zeros(height(allPreTestSAL{jj}),1); ...
        %             ones(height(allPreTestSAL{jj}),1)];
        % 
        %         % Ind
        %         scores = predict(salMdlComb,thisWashTestXsal);
        %         [~,~,~,salA_glm80_comb_washInd(jj,ii,k)] = perfcurve( ...
        %             thisWashTestYsal,scores,1);
        %         scores = predict(salMdlPermComb,thisWashTestXsal);
        %         [~,~,~,salA_glm80_comb_washInd_perm(jj,ii,k)] = perfcurve( ...
        %             thisWashTestYsal,scores,1);
        %     else
        %         salA_glm80_comb_washInd(jj,ii,k) = NaN;
        %         salA_glm80_comb_washInd_perm(jj,ii,k) = NaN;
        %     end
        % end
        % % LSD
        % for jj = 1:numel(lsd_ids_all)
        %     if ~isempty(allWashLSD{jj,ii})
        %         thisWashTestXlsd = [allPreTestLSD{jj}; ...
        %             allWashLSD{jj,ii}.normData(randperm(height( ...
        %             allWashLSD{jj,ii}),min([height( ...
        %             allWashLSD{jj,ii}) height(allPostTestLSD{jj})])),:)];
        % 
        %         thisWashTestYlsd = [zeros(height(allPreTestLSD{jj}),1); ...
        %             ones(height(allPreTestLSD{jj}),1)];
        % 
        %         % Ind
        %         scores = predict(lsdMdlComb,thisWashTestXlsd);
        %         [~,~,~,lsdA_glm80_comb_washInd(jj,ii,k)] = perfcurve( ...
        %             thisWashTestYlsd,scores,1);
        %         scores = predict(lsdMdlPermComb,thisWashTestXlsd);
        %         [~,~,~,lsdA_glm80_comb_washInd_perm(jj,ii,k)] = perfcurve( ...
        %             thisWashTestYlsd,scores,1);
        %     else
        %         lsdA_glm80_comb_washInd(jj,ii,k) = NaN;
        %         lsdA_glm80_comb_washInd_perm(jj,ii,k) = NaN;
        %     end
        % end

        scores = predict(salMdlComb,cat(1,washTestXsal1{:,ii}, ...
            washTestXsal2{:,ii}));
        [~,~,~,salA_glm80_comb_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii},washTestYsal2{:,ii}),scores,1);

        scores = predict(lsdMdlComb,cat(1,washTestXlsd1{:,ii}, ...
            washTestXlsd2{:,ii}));
        [~,~,~,lsdA_glm80_comb_wash(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii},washTestYlsd2{:,ii}),scores,1);

        scores = predict(salMdlStimComb,cat(1,washTestXsal1{:,ii}, ...
            washTestXsal2{:,ii}));
        [~,~,~,salA_glm80_comb_wash_stim(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii},washTestYsal2{:,ii}),scores,1);

        scores = predict(lsdMdlStimComb,cat(1,washTestXlsd1{:,ii}, ...
            washTestXlsd2{:,ii}));
        [~,~,~,lsdA_glm80_comb_wash_stim(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii},washTestYlsd2{:,ii}),scores,1);

        scores = predict(salMdlPermComb,cat(1,washTestXsal1{:,ii}, ...
            washTestXsal2{:,ii}));
        [~,~,~,salA_glm80_comb_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestYsal1{:,ii},washTestYsal2{:,ii}),scores,1);

        scores = predict(lsdMdlPermComb,cat(1,washTestXlsd1{:,ii}, ...
            washTestXlsd2{:,ii}));
        [~,~,~,lsdA_glm80_comb_wash_perm(ii,k)] = perfcurve(cat(1, ...
            washTestYlsd1{:,ii},washTestYlsd2{:,ii}),scores,1);

        % % Cross applied
        % scores = predict(lsdMdlComb,cat(1,washTestXsal1{:,ii}, ...
        %     washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_comb_wash_lsdTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYsal1{:,ii},washTestYsal2{:,ii}), ...
        %     scores,1);
        % scores = predict(salMdlComb,cat(1,washTestXlsd1{:,ii}, ...
        %     washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_comb_wash_salTrain(ii,k)] = perfcurve(cat(1, ...
        %     washTestYlsd1{:,ii},washTestYlsd2{:,ii}), ...
        %     scores,1);
        % scores = predict(lsdMdlStimComb,cat(1,washTestXsal1{:,ii}, ...
        %     washTestXsal2{:,ii}));
        % [~,~,~,salA_glm80_comb_stim_wash_lsdTrain(ii,k)] = ...
        %     perfcurve(cat(1,washTestYsal1{:,ii},washTestYsal2{:,ii}), ...
        %     scores,1);
        % scores = predict(salMdlStimComb,cat(1,washTestXlsd1{:,ii}, ...
        %     washTestXlsd2{:,ii}));
        % [~,~,~,lsdA_glm80_comb_stim_wash_salTrain(ii,k)] = ...
        %     perfcurve(cat(1,washTestYlsd1{:,ii},washTestYlsd2{:,ii}), ...
        %     scores,1);

        % Single features
        lsdX = cat(1,washTestXlsd2{:,ii},washTestXlsd1{:,ii});
        lsdY = cat(1,washTestYlsd2{:,ii},washTestYlsd1{:,ii});
        salX = cat(1,washTestXsal2{:,ii},washTestXsal1{:,ii});
        salY = cat(1,washTestYsal2{:,ii},washTestYsal1{:,ii});
        for f = 1:216
            scores = predict(salMdlSingleComb{f},salX(:,f));
            [~,~,~,salA_glm80_comb_wash_single(ii,f,k)] = ...
                perfcurve(salY,scores,1);
            scores = predict(salMdlSinglePermComb{f},salX(:,f));
            [~,~,~,salA_glm80_comb_wash_single_perm(ii,f,k)] = ...
                perfcurve(salY,scores,1);

            scores = predict(lsdMdlSingleComb{f},lsdX(:,f));
            [~,~,~,lsdA_glm80_comb_wash_single(ii,f,k)] = ...
                perfcurve(lsdY,scores,1);
            scores = predict(lsdMdlSinglePermComb{f},lsdX(:,f));
            [~,~,~,lsdA_glm80_comb_wash_single_perm(ii,f,k)] = ...
                perfcurve(lsdY,scores,1);

            % scores = predict(salMdlSingleCombStim,salX(:,f));
            % [~,~,~,salA_glm80_comb_stim_wash_single(ii,f,k)] = ...
            %     perfcurve(salY,scores,1);
            % scores = predict(salMdlSinglePermCombStim{f},salX(:,f));
            % [~,~,~,salA_glm80_comb_stim_wash_single_perm(ii,f,k)] = ...
            %     perfcurve(salY,scores,1);

            scores = predict(lsdMdlSingleCombStim,lsdX(:,f));
            [~,~,~,lsdA_glm80_comb_stim_wash_single(ii,f,k)] = ...
                perfcurve(lsdY,scores,1);
            % scores = predict(lsdMdlSinglePermCombStim{f},lsdX(:,f));
            % [~,~,~,lsdA_glm80_comb_stim_wash_single_perm(ii,f,k)] = ...
            %     perfcurve(lsdY,scores,1);
        end
    end    
end
%% Save
save('H:\LSD+stim_persist\publicationData\modelOutputs.mat', ...
    'lsdA_glm80_comb1_wash','lsdA_glm80_comb1_wash_stim', ...
    'salA_glm80_comb1_wash','salA_glm80_comb1_wash_stim', ...
    'lsdA_glm80_1hr_wash','salA_glm80_1hr_wash', ...
    'salA_glm80_comb1_wash_perm','lsdA_glm80_comb1_wash_perm', ...
    'salA_glm80_1hr_wash_single','lsdA_glm80_1hr_wash_single', ...
    'lsdA_glm80_comb_wash','salA_glm80_comb_wash', ...
    'lsdA_glm80_comb_wash_stim','salA_glm80_comb_wash_stim', ...
    'lsdA_glm80_2hr_wash','salA_glm80_2hr_wash', ...
    'salA_glm80_2hr_wash_perm','lsdA_glm80_2hr_wash_perm', ...
    'salA_glm80_comb_wash_perm','lsdA_glm80_comb_wash_perm', ...
    'salA_glm80_comb_wash_single_perm','lsdA_glm80_comb_wash_single_perm', ...
    'lsdA_glm80_comb_postBinInd','salA_glm80_comb_postBinInd', ...
    'lsdA_glm80_comb_wash_single','lsdA_glm80_2hr_wash_single', ...
    'salA_glm80_comb_wash_single','salA_glm80_2hr_wash_single', ...
    'salA_glm80_comb_stim_wash_single','lsdA_glm80_comb_stim_wash_single', ...
    'lsdA_glm80_comb_single','salA_glm80_comb_single', ...
    'postSignLSDcombStim','lsdA_glm80_comb_single_stim', ...
    'postSignSALcomb','postSignLSDcomb','postSignPermSALcomb', ...
    'postSignPermLSDcomb','salA_glm80_comb_single_perm', ...
    'lsdA_glm80_comb_single_perm','postSignSAL1','postSignSAL2', ... 
    'postSignLSD1','postSignLSD2','salA_glm80_1hr_single', ...
    'salA_glm80_2hr_single','lsdA_glm80_1hr_single', ...
    'lsdA_glm80_2hr_single','postSignPermSAL1','postSignPermSAL2', ...
    'postSignPermLSD1','postSignPermLSD2','salA_glm80_1hr_single_perm', ...
    'lsdA_glm80_1hr_single_perm', ...
    'lsdA_glm80_comb_salTrain', ...
    'salA_glm80_comb_lsdTrain','lsdVsalA_glm80_stim_comb', ...
    'lsdVsalA_glm80_post_comb','lsdVsalA_glm80_stim_comb_perm', ...
    'lsdVsalA_glm80_post_comb_perm','lsdA_glm80_stim_comb', ...
    'lsdA_glm80_comb','lsdA_glm80_comb_stim_post', ...
    'salA_glm80_stim_comb','salA_glm80_comb','salA_glm80_comb_stim_post', ...
    'lsdA_glm80_1hr_perm','salA_glm80_1hr_perm','lsdA_glm80_2hr', ... 
    'lsdA_glm80_2hr_perm','salA_glm80_2hr_perm','lsdA_glm80_1hr_salTrain', ...
    'salA_glm80_1hr_lsdTrain','lsdA_glm80_2hr_salTrain', ...
    'salA_glm80_2hr_lsdTrain','lsdA_glm80_1hr_stim','lsdA_glm80_1hr', ...
    'lsdA_glm80_1hr_stim_post','salA_glm80_1hr_stim','salA_glm80_1hr', ...
    'salA_glm80_1hr_stim_post','lsdA_glm80_2hr_stim', ...
    'lsdA_glm80_2hr_stim_post','salA_glm80_2hr_stim', ...
    'salA_glm80_2hr','salA_glm80_2hr_stim_post','salA_glm80_comb_perm', ...
    'lsdA_glm80_comb_perm','salA_glm80_comb_m','lsdA_glm80_comb_m', ...
    'salA_glm80_comb_f','lsdA_glm80_comb_f');
%%
% 'salA_glm80_2hr_single_perm','lsdA_glm80_2hr_single_perm',
%% LSD vs. SAL combined cohorts (Figure 2)
load('H:\LSD+stim_persist\publicationData\modelOutputs.mat', ...
    'lsdA_glm80_comb','salA_glm80_comb','lsdA_glm80_comb_wash', ...
    'salA_glm80_comb_wash','lsdA_glm80_comb1_wash', ...
    'salA_glm80_comb1_wash','lsdA_glm80_comb_postBinInd', ...
    'salA_glm80_comb_postBinInd')
load('H:\LSD+stim_persist\publicationData\shamStimMdl.mat', ...
    'baseA_glm80_wash')
figure
subplot(3,2,1)
violinScatter({lsdA_glm80_comb,salA_glm80_comb},...
    'colors',{'r','b'}','labels',...
    {'LSD','SAL'},'xvals',[1,1.5])
title('combined: pre vs. post')
p = permutationTest(lsdA_glm80_comb,salA_glm80_comb,'+');
sigstar({[1,1.5]},p);
ylabel('AUC')
box off
ylim([0.5 0.85])

subplot(3,2,2)
boxplot(lsdA_glm80_comb_wash','colors','r','notch',1,'widths',0.25, ...
    'symbol','or')
hold on
boxplot(salA_glm80_comb_wash','colors','b','notch',1,'widths',0.25,...
    'positions',1.25:1:2.25,'symbol','ob')
boxplot(lsdA_glm80_comb1_wash(3:5,:)','colors','r','notch',1, ...
    'widths',0.25, ...
    'positions',3:5,'symbol','or')
boxplot(salA_glm80_comb1_wash(3:5,:)','colors','b','notch',1, ...
    'widths',0.25, ...
    'symbol','or','positions',3.25:5.25)

yline(mean(baseA_glm80_wash,'all'),'--k','linewidth',2)

p = NaN(1,5);
p(1) = permutationTest(lsdA_glm80_comb_wash(1,:), ...
    salA_glm80_comb_wash(1,:),'+');
p(2) = permutationTest(lsdA_glm80_comb_wash(2,:), ...
    salA_glm80_comb_wash(2,:),'+');
p(3) = permutationTest(lsdA_glm80_comb1_wash(3,:), ...
    salA_glm80_comb1_wash(3,:),'+');
p(4) = permutationTest(lsdA_glm80_comb1_wash(4,:), ...
    salA_glm80_comb1_wash(4,:),'+');
p(5) = permutationTest(lsdA_glm80_comb1_wash(5,:), ...
    salA_glm80_comb1_wash(5,:),'+');
sigstar({[1,1.25],[2,2.25],[3,3.25],[4,4.25],[5,5.25]},p)
set(gca,'xtick',1:5,'xticklabel',{'24 hr','48 hr','72 hr','144 hr', ...
    '168 hr'})
box off
ylabel('AUC')
ylim([0.5 0.85])

subplot(3,2,3)
plot(mean(lsdA_glm80_comb_postBinInd,3)','.r')
ylim([0.4 1])
box off
set(gca,'xtick',1:12,'xticklabel',5:5:60)
xlabel('time from stim (min)')
ylabel('AUC')

subplot(3,2,4)
plot(mean(salA_glm80_comb_postBinInd,3)','.b')
ylim([0.4 1])
box off
set(gca,'xtick',1:12,'xticklabel',5:5:60)
xlabel('time from stim (min)')
ylabel('AUC')

subplot(3,2,5)
violinScatter({std(mean(lsdA_glm80_comb_postBinInd,3,'omitnan'),[],2, ...
    'omitnan'),std(mean(salA_glm80_comb_postBinInd,3,'omitnan'),[],2, ...
    'omitnan')},'colors',{'r';'b'})
[~,p] = ttest2(std(mean(lsdA_glm80_comb_postBinInd,3,'omitnan'), ...
    [],2,'omitnan'),std(mean(salA_glm80_comb_postBinInd,3,'omitnan'), ...
    [],2,'omitnan'));
sigstar([1,2],p)
box off
title('variability across time')
ylabel('AUC STD')
set(gca,'xticklabel',{'LSD','SAL'})

subplot(3,2,6)
title('variability across rat')
violinScatter({std(mean(lsdA_glm80_comb_postBinInd,3,'omitnan'),[],1, ...
    'omitnan'), std(mean(salA_glm80_comb_postBinInd,3,'omitnan'),[],1, ...
    'omitnan')},'colors',{'r';'b'})
[~,p] = ttest2(std(mean(lsdA_glm80_comb_postBinInd,3,'omitnan'), ...
    [],1,'omitnan')',std(mean(salA_glm80_comb_postBinInd,3,'omitnan'), ...
    [],1,'omitnan')');
sigstar([1,2],p)
box off
ylabel('AUC STD')
set(gca,'xticklabel',{'LSD','SAL'})
%% SAL vs LSD (Figure 3)
load('H:\LSD+stim_persist\publicationData\modelOutputs.mat', ...
    'lsdA_glm80_comb_wash_single','lsdA_glm80_1hr_wash_single', ...
    'salA_glm80_comb_wash_single','salA_glm80_1hr_wash_single', ...
    'lsdA_glm80_comb_single_perm','salA_glm80_comb_single_perm', ...
    'lsdA_glm80_comb_single','salA_glm80_comb_single','lsdA_glm80_comb', ...
    'lsdA_glm80_comb_salTrain','salA_glm80_comb_lsdTrain', ...
    'salA_glm80_comb','lsdVsalA_glm80_stim_comb', ...
    'lsdVsalA_glm80_post_comb','lsdVsalA_glm80_stim_comb_perm', ...
    'lsdVsalA_glm80_post_comb_perm','lsdA_glm80_comb_wash_single_perm', ...
    'salA_glm80_comb_wash_single_perm','postSignSALcomb','postSignLSDcomb');

[lsdCombWashSingle,indLSD] = sort(mean( ...
    lsdA_glm80_comb_wash_single(1,:,:),3)','descend'); %#ok<*UDIM>
[salCombWashSingle,indSAL] = sort( ...
    mean(salA_glm80_comb_wash_single(1,:,:),3)','descend');

lsdPerm = squeeze(lsdA_glm80_comb_wash_single_perm(1,:,:));
lsdPermThresh = prctile(reshape(lsdPerm,1,[]),95);

inds = find(mean(lsdA_glm80_comb_wash_single(1,:,:),3)'>lsdPermThresh);
[p,m] = deal(NaN(numel(inds),1));
[lsd,sal] = deal(NaN(numel(inds),2));
for ii = 1:numel(inds)
    [p(ii)] = permutationTest(lsdA_glm80_comb_wash_single(1,inds(ii),:), ...
        salA_glm80_comb_wash_single(1,inds(ii),:),'+');
    m(ii) = sum(mean(lsdA_glm80_comb_wash_single(1,inds(ii),:))< ...
        salA_glm80_comb_wash_single(1,inds(ii),:))+1;
    lsd(ii,:) = [mean(lsdA_glm80_comb_wash_single(1,inds(ii),:)) 
        std(lsdA_glm80_comb_wash_single(1,inds(ii),:))];
    sal(ii,:) = [mean(salA_glm80_comb_wash_single(1,inds(ii),:)) 
        std(salA_glm80_comb_wash_single(1,inds(ii),:))];
end

% Get feature IDs
feat = names({'lmPFC','rmPFC','lOFC','rOFC','lNAcS','rNAcS','lNAcC',...
    'rNAcC'},{'d','t','a','b','lg','hg'})';
lsdFeat = table(indLSD,feat(indLSD),lsdCombWashSingle.* ...
    sign(mean(postSignLSDcomb(indLSD,:),2)));
salFeat = table(indSAL,feat(indSAL),salCombWashSingle.* ...
    sign(mean(postSignSALcomb(indSAL,:),2)));

figure
histogram(lsdPerm,'normalization','probability')
xline(lsdPermThresh,'--r')
box off
xlabel('AUC')
title('LSD permuted AUC distribution')

figure
subplot(2,5,1:2)
hold on
plot(1:216,lsdCombWashSingle,'.r')
plot(1:216,mean(salA_glm80_comb_wash_single(:,indLSD,:),[1,3])','.b')
yline(lsdPermThresh,'--r')
xlabel('ranked feature')
ylabel('AUC')
xlim([0 216])
title('LSD and SAL pre vs. post -> wash')
box off

subplot(2,5,5)
thisLSD = [mean(lsdA_glm80_comb_single,2)';
    mean(lsdA_glm80_comb_wash_single,3)];
thisSAL = [mean(salA_glm80_comb_single,2)';
    mean(salA_glm80_comb_wash_single,3)];
inds = inds(p<0.05);
hold on
plot(thisLSD(1:2,inds),'-or')
set(gca,'xtick',1:2,'xticklabel',{'pre vs. post','24 hours'})
plot(thisSAL(1:2,inds),'-ob')
xlim([0.9 2.1])
set(gca,'xtick',1:2,'xticklabel',{'pre vs. post','24 hours'})

subplot(2,2,3)
violinScatter({lsdA_glm80_comb,lsdA_glm80_comb_salTrain,...
    salA_glm80_comb_lsdTrain,salA_glm80_comb},'colors',{'r';'c';'m';'b'}, ...
    'labels',{'LSD\rightarrowLSD','SAL\rightarrowLSD','LSD\rightarrowSAL',...
    'SAL\rightarrowSAL'})
ylim([0.45 0.9])
title('combined cross-applied pre v. post: train\rightarrowtest')
ylabel('AUC')
box off

subplot(2,2,4)
violinScatter({lsdVsalA_glm80_stim_comb,lsdVsalA_glm80_post_comb}, ...
    'colors',{'g';'m'})
hold on
violinScatter({lsdVsalA_glm80_stim_comb_perm, ...
    lsdVsalA_glm80_post_comb_perm})
set(gca,'xticklabel',{'stim','post'})
ylabel('AUC')
title('LSD vs. SAL')
box off
%% Stim vs post - combined cohorts (Figure 4A-B)
load('H:\LSD+stim_persist\publicationData\modelOutputs.mat', ...
    'lsdA_glm80_stim_comb','lsdA_glm80_comb',...
    'lsdA_glm80_comb_stim_post','salA_glm80_stim_comb','salA_glm80_comb', ...
    'salA_glm80_comb_stim_post','lsdA_glm80_comb_wash', ...
    'lsdA_glm80_comb_wash_stim','salA_glm80_comb_wash', ...
    'salA_glm80_comb_wash_stim','lsdA_glm80_comb1_wash', ...
    'lsdA_glm80_comb1_wash_stim','salA_glm80_comb1_wash', ...
    'salA_glm80_comb1_wash_stim')
figure
subplot(1,3,1)
violinScatter({lsdA_glm80_stim_comb,lsdA_glm80_comb,...
    lsdA_glm80_comb_stim_post,salA_glm80_stim_comb,salA_glm80_comb,...
    salA_glm80_comb_stim_post},'colors',{[0.6479, 0.2000, 0.3714]; ...
    [1 0 0];[1.0000, 0.7414, 0.8965];[0.0860, 0.1908, 0.6159];[0 0 1]; ...
    [0.6028, 0.7485, 1.0000]})
ylim([0.4 0.9])
box off
set(gca,'xticklabel',{'S->S','P->P','S->P','S->S','P->P','S->P'})
permutationTest(lsdA_glm80_stim_comb,lsdA_glm80_comb,'+')

% Stim vs post
subplot(1,3,2:3)
hold on
boxplot(lsdA_glm80_comb_wash','colors','r','notch',1,'widths',1, ...
    'symbol','.r','positions',1:7:8)
boxplot(lsdA_glm80_comb_wash_stim','colors',[0.6479, 0.2000, 0.3714], ...
    'notch',1,'widths',1,'symbol','.','positions',2:7:9)
boxplot(salA_glm80_comb_wash','colors','b','notch',1,'widths',1, ...
    'symbol','.b','positions',4:7:11)
boxplot(salA_glm80_comb_wash_stim','colors',[0.0860, 0.1908, 0.6159], ...
    'notch',1,'widths',1,'symbol','.','positions',5:7:12)

boxplot(lsdA_glm80_comb1_wash(3:5,:)','colors','r','notch',1,'widths',1, ...
    'symbol','.r','positions',15:7:29)
boxplot(lsdA_glm80_comb1_wash_stim(3:5,:)','colors', ...
    [0.6479, 0.2000, 0.3714],'notch',1,'widths',1,'symbol','.', ...
    'positions',16:7:30)
boxplot(salA_glm80_comb1_wash(3:5,:)','colors','b','notch',1,'widths',1, ...
    'symbol','.b','positions',18:7:32)
boxplot(salA_glm80_comb1_wash_stim(3:5,:)','colors', ...
    [0.0860, 0.1908, 0.6159],'notch',1,'widths',1,'symbol','.', ...
    'positions',19:7:33)
box off
ylim([0.4 0.9])
xlim([0.4 33.6])
set(gca,'xtick',3:7:31,'xticklabel',{'24 hr','48 hr','72 hr','144 hr','168 hr'})

this = [lsdA_glm80_comb_wash;lsdA_glm80_comb1_wash(3:5,:);...
    lsdA_glm80_comb_wash_stim;lsdA_glm80_comb1_wash_stim(3:5,:)];
auc = mean(this,2);
time = repmat(1:5,1,2)';
group = [0;0;0;0;0;1;1;1;1;1];
tbl = table(auc,group,time);
lsdLME = fitglme(tbl,'auc~group*time');

this = [salA_glm80_comb_wash;salA_glm80_comb1_wash(3:5,:);...
    salA_glm80_comb_wash_stim;salA_glm80_comb1_wash_stim(3:5,:)];
auc = mean(this,2);
time = repmat(1:5,1,2)';
group = [0;0;0;0;0;1;1;1;1;1];
tbl = table(auc,group,time);
salLME = fitglme(tbl,'auc~group*time');
%% LSD single features stim vs. wash (Figure 4D)
load('H:\LSD+stim_persist\publicationData\modelOutputs.mat', ...
    'lsdA_glm80_comb_single_stim','lsdA_glm80_comb_stim_wash_single')
figure
plot(mean(lsdA_glm80_comb_single_stim,2), ...
    mean(lsdA_glm80_comb_stim_wash_single, ...
    [1,3])','.r')
lsline
[r,p] = corrcoef(mean(lsdA_glm80_comb_stim_wash_single(1,:,:),3)', ...
    mean(lsdA_glm80_comb_single_stim,2));
title(['LSD stim: ','r = ',num2str(round(r(1,2),2)),'; p = ', ...
    num2str(round(p(1,2),2))])
ylabel('stim->wash AUC')
xlabel('stim->stim AUC')
box off
%% mTOR and PNN data (Figure 5)
% Load data
load('H:\LSD+stim_persist\publicationData\pS6_PNN.mat')

% Group by brain region
regionsIL = {'_IL1_','_IL2_','_IL3_','_IL4_'};
regionsPL = {'_mPFC1_','_mPFC2_','_mPFC3_'}; % mPFC = PL
regionsDG = {'_DG1_','_DG2_'};

% Set up rat ID by group
LSD_sIL_ID = {'LSD10_','LSD14_','LSD20_','IRDM124_'};
LSD_sham_ID = {'LSD11_','IRDM136_','IRDM141_','LSD19_','LSD23_','IRDM118_'};
SAL_sham_ID = {'LSD12_','LSD13_','IRDM140_','LSD17_','LSD21_'};
SAL_sIL_ID = {'LSD15_','LSD09_','IRDM139_','LSD18_','LSD22_','IRDM125_'};
% Combine all IDs
allID = cat(2,SAL_sIL_ID,SAL_sham_ID,LSD_sham_ID,LSD_sIL_ID);

% Set up pS6 data
pS6.group(contains(pS6.ID,LSD_sIL_ID)) = {'LSD-sIL'};
pS6.group(contains(pS6.ID,LSD_sham_ID)) = {'LSD-sham'};
pS6.group(contains(pS6.ID,SAL_sIL_ID)) = {'SAL-sIL'};
pS6.group(contains(pS6.ID,SAL_sham_ID)) = {'SAL-sham'};
% Truncate ID
for ii = 1:numel(allID)
    pS6.IDshort(contains(pS6.ID,allID{ii})) = allID(ii);
end

% Set up PNN data
PNN.group(contains(PNN.ID,LSD_sIL_ID)) = {'LSD-sIL'};
PNN.group(contains(PNN.ID,LSD_sham_ID)) = {'LSD-sham'};
PNN.group(contains(PNN.ID,SAL_sIL_ID)) = {'SAL-sIL'};
PNN.group(contains(PNN.ID,SAL_sham_ID)) = {'SAL-sham'};
% Truncate ID
for ii = 1:numel(allID)
    PNN.IDshort(contains(PNN.ID,allID{ii})) = allID(ii);
end

% Add in age (z-scored and raw days) and sex data
for ii = 1:size(demog,1)
    pS6.ageZ(contains(pS6.IDshort,demog{ii,1})) = demog{ii,4};
    pS6.age(contains(pS6.IDshort,demog{ii,1})) = demog{ii,3};
    pS6.sex(contains(pS6.IDshort,demog{ii,1})) = demog{ii,2};

    PNN.ageZ(contains(PNN.IDshort,demog{ii,1})) = demog{ii,4};
    PNN.age(contains(PNN.IDshort,demog{ii,1})) = demog{ii,3};
    PNN.sex(contains(PNN.IDshort,demog{ii,1})) = demog{ii,2};
end
% IL - pS6
% LME
lme_pS6_IL = fitglme(pS6(contains(pS6.ID,regionsIL),:),...
    'mean ~ sex+ageZ+group+(1|IDshort)','distribution','gamma','link','log');
stats_pS6_IL = anova(lme_pS6_IL);

% Post-hoc group comparisons
% Preallocate
pS6_IL = NaN(1,6);
% Contrasts
L = [0 1 0 0 0 0; ... % LSD-sIL vs LSD-sham
    0 0 1 0 0 0; ...% SAL-sIL vs LSD-sham
    0 0 0 1 0 0; ...% SAL-sham vs LSD-sham
    0 -1 1 0 0 0; ... % SAL-sIL vs LSD-sIL
    0 -1 0 1 0 0;...% SAL-sham vs LSD-sIL
    0 0 -1 1 0 0];    % SAL-sham vs SAL-sIL
for ii = 1:6
    pS6_IL(ii) = coefTest(lme_pS6_IL, L(ii,:));
end
% PL - pS6
% LME
lme_pS6_PL = fitglme(pS6(contains(pS6.ID,regionsPL),:),...
    'mean ~ sex+ageZ+group','distribution','gamma','link','log');
stats_pS6_mPFC = anova(lme_pS6_PL);

% Post-hoc comparisons
% Preallocate
pS6_pl = NaN(1,6);
% Define contrasts
L = [0 1 0 0 0 0 ; ... % LSD-sIL vs LSD-sham
    0 0 1 0 0 0; ...% SAL-sIL vs LSD-sham
    0 0 0 1 0 0; ...% SAL-sham vs LSD-sham
    0 -1 1 0 0 0; ... % SAL-sIL vs LSD-sIL
    0 -1 0 1 0 0;...% SAL-sham vs LSD-sIL
    0 0 -1 1 0 0];    % SAL-sham vs SAL-sIL
for ii = 1:6
    pS6_pl(ii) = coefTest(lme_pS6_PL, L(ii,:));
end
% DG - pS6
% LME
lme_pS6_DG = fitglme(pS6(contains(pS6.ID,regionsDG),:),...
    'mean ~ sex+ageZ+group','distribution','gamma','Link','log');
stats_pS6_DG = anova(lme_pS6_DG);

% Post-hoc group comparisons
% Preallocate
pS6_DG = NaN(1,6);
% Define contrasts
L = [0 1 0 0 0 0; ... % LSD-sIL vs LSD-sham
    0 0 1 0 0 0 ; ...% SAL-sIL vs LSD-sham
    0 0 0 1 0 0 ; ...% SAL-sham vs LSD-sham
    0 -1 1 0 0 0 ; ... % SAL-sIL vs LSD-sIL
    0 -1 0 1 0 0 ;...% SAL-sham vs LSD-sIL
    0 0 -1 1 0 0 ];    % SAL-sham vs SAL-sIL
for ii = 1:6
    pS6_DG(ii) = coefTest(lme_pS6_DG, L(ii,:));
end
% IL - PNN
% LME
lme_PNN_IL = fitglme(PNN(contains(PNN.ID,regionsIL),:),...
    'mean ~ sex+age+group','distribution','gamma','link','log');
stats_PNN_IL = anova(lme_PNN_IL);

% Post-hoc group comparisons
% Define contrasts
L = [0 1 0 0 0 0; ... % LSD-sIL vs LSD-sham
    0 0 1 0 0 0; ...% SAL-sIL vs LSD-sham
    0 0 0 1 0 0; ...% SAL-sham vs LSD-sham
    0 -1 1 0 0 0; ... % SAL-sIL vs LSD-sIL
    0 -1 0 1 0 0;...% SAL-sham vs LSD-sIL
    0 0 -1 1 0 0];    % SAL-sham vs SAL-sIL
% Preallocate
PNN_IL = NaN(1,6);
for ii = 1:6
    PNN_IL(ii) = coefTest(lme_PNN_IL, L(ii,:));
end
% PL - PNN
% LME
lme_PNN_mPFC = fitglme(PNN(contains(PNN.ID,regionsPL),:),...
    'mean ~ sex+age+group','distribution','gamma','link','log');
stats_PNN_mPFC = anova(lme_PNN_mPFC);

% Post-hoc group comparisons
% Preallocate
PNN_mpfc = NaN(1,6);
% Define contrasts
L = [0 1 0 0 0 0; ... % LSD-sIL vs LSD-sham
    0 0 1 0 0 0; ...% SAL-sIL vs LSD-sham
    0 0 0 1 0 0; ...% SAL-sham vs LSD-sham
    0 -1 1 0 0 0; ... % SAL-sIL vs LSD-sIL
    0 -1 0 1 0 0;...% SAL-sham vs LSD-sIL
    0 0 -1 1 0 0];    % SAL-sham vs SAL-sIL
for ii = 1:6
    PNN_mpfc(ii) = coefTest(lme_PNN_mPFC, L(ii,:));
end
% DG - PNN
% LME
lme_PNN_DG = fitglme(PNN(contains(PNN.ID,regionsDG),:),...
    'mean ~ sex+age+group','distribution','gamma','link','log');
stats_PNN_DG = anova(lme_PNN_DG);

% Post-hoc group comparisons
% Preallocate
PNN_DG = NaN(1,6);
% Define contrasts
L = [0 1 0 0 0 0; ... % LSD-sIL vs LSD-sham
    0 0 1 0 0 0; ...% SAL-sIL vs LSD-sham
    0 0 0 1 0 0; ...% SAL-sham vs LSD-sham
    0 -1 1 0 0 0; ... % SAL-sIL vs LSD-sIL
    0 -1 0 1 0 0;...% SAL-sham vs LSD-sIL
    0 0 -1 1 0 0];    % SAL-sham vs SAL-sIL
for ii = 1:6
    PNN_DG(ii) = coefTest(lme_PNN_DG, L(ii,:));
end

% Plot all (Figure 5 A-F)
figure
tlo = tiledlayout(2,3);
% pS6 IL
a = contains(pS6.ID,LSD_sIL_ID) & contains(pS6.ID,regionsIL);
b = contains(pS6.ID,LSD_sham_ID) & contains(pS6.ID,regionsIL);
c = contains(pS6.ID,SAL_sIL_ID) & contains(pS6.ID,regionsIL);
d = contains(pS6.ID,SAL_sham_ID) & contains(pS6.ID,regionsIL);
ax = nexttile(tlo);
hold on
plotSpread(ax,[{pS6.mean(a)},{pS6.mean(b)},{pS6.mean(c)},{pS6.mean(d)}])
boxplot(ax,pS6.mean,pS6.group,'notch',1,'symbol','')
title('IL')
ylabel('pS6')
set(gca,'xticklabel',{'LSD-sIL','LSD-sham','SAL-sIL','SAL-sham'})
ylim([0 80])
box off

% pS6 mPFC
a = contains(pS6.ID,LSD_sIL_ID) & contains(pS6.ID,regionsPL);
b = contains(pS6.ID,LSD_sham_ID) & contains(pS6.ID,regionsPL);
c = contains(pS6.ID,SAL_sIL_ID) & contains(pS6.ID,regionsPL);
d = contains(pS6.ID,SAL_sham_ID) & contains(pS6.ID,regionsPL);
ax = nexttile(tlo);
hold on
plotSpread(ax,[{pS6.mean(a)},{pS6.mean(b)},{pS6.mean(c)},{pS6.mean(d)}])
boxplot(ax,pS6.mean,pS6.group,'notch',1,'symbol','')
title('mPFC')
ylabel('pS6')
set(gca,'xticklabel',{'LSD-sIL','LSD-sham','SAL-sIL','SAL-sham'})
ylim([0 80])
box off

% pS6 DG
a = contains(pS6.ID,LSD_sIL_ID) & contains(pS6.ID,regionsDG);
b = contains(pS6.ID,LSD_sham_ID) & contains(pS6.ID,regionsDG);
c = contains(pS6.ID,SAL_sIL_ID) & contains(pS6.ID,regionsDG);
d = contains(pS6.ID,SAL_sham_ID) & contains(pS6.ID,regionsDG);
ax = nexttile(tlo);
hold on
plotSpread(ax,[{pS6.mean(a)},{pS6.mean(b)},{pS6.mean(c)},{pS6.mean(d)}])
boxplot(ax,pS6.mean,pS6.group,'notch',1,'symbol','')
title('DG')
ylabel('pS6')
set(gca,'xticklabel',{'LSD-sIL','LSD-sham','SAL-sIL','SAL-sham'})
ylim([0 80])
box off

% PNN IL
a = contains(PNN.ID,LSD_sIL_ID) & contains(PNN.ID,regionsIL);
b = contains(PNN.ID,LSD_sham_ID) & contains(PNN.ID,regionsIL);
c = contains(PNN.ID,SAL_sIL_ID) & contains(PNN.ID,regionsIL);
d = contains(PNN.ID,SAL_sham_ID) & contains(PNN.ID,regionsIL);
ax = nexttile(tlo);
hold on
plotSpread(ax,[{PNN.mean(a)},{PNN.mean(b)},{PNN.mean(c)},{PNN.mean(d)}])
boxplot(ax,PNN.mean,PNN.group,'notch',1,'symbol','')
title('IL')
ylabel('PNN')
set(gca,'xticklabel',{'LSD-sIL','LSD-sham','SAL-sIL','SAL-sham'})
ylim([40 160])
box off

% PNN mPFC
a = contains(PNN.ID,LSD_sIL_ID) & contains(PNN.ID,regionsPL);
b = contains(PNN.ID,LSD_sham_ID) & contains(PNN.ID,regionsPL);
c = contains(PNN.ID,SAL_sIL_ID) & contains(PNN.ID,regionsPL);
d = contains(PNN.ID,SAL_sham_ID) & contains(PNN.ID,regionsPL);
ax = nexttile(tlo);
hold on
plotSpread(ax,[{PNN.mean(a)},{PNN.mean(b)},{PNN.mean(c)},{PNN.mean(d)}])
boxplot(ax,PNN.mean,PNN.group,'notch',1,'symbol','')
title('mPFC')
ylabel('PNN')
set(gca,'xticklabel',{'LSD-sIL','LSD-sham','SAL-sIL','SAL-sham'})
ylim([40 160])
box off

% PNN DG
a = contains(PNN.ID,LSD_sIL_ID) & contains(PNN.ID,regionsDG);
b = contains(PNN.ID,LSD_sham_ID) & contains(PNN.ID,regionsDG);
c = contains(PNN.ID,SAL_sIL_ID) & contains(PNN.ID,regionsDG);
d = contains(PNN.ID,SAL_sham_ID) & contains(PNN.ID,regionsDG);
ax = nexttile(tlo);
hold on
plotSpread(ax,[{PNN.mean(a)},{PNN.mean(b)},{PNN.mean(c)},{PNN.mean(d)}])
boxplot(ax,PNN.mean,PNN.group,'notch',1,'symbol','')
title('DG')
ylabel('PNN')
set(gca,'xticklabel',{'LSD-sIL','LSD-sham','SAL-sIL','SAL-sham'})
ylim([40 160])
box off
%% Sham stimulation data (Extended Data Figure 1)
% Load data
load('H:\LSD+stim_persist\publicationData\shamStimMdl.mat')
figure
violinScatter([{baseA_glm80},{baseA_glm80_wash(1,:)}, ...
    {baseA_glm80_wash(2,:)},{baseA_glm80_wash(3,:)}],'labels', ...
    {'base','24 hr','48 hr','96 hr'})
ylabel('AUC')
%% Sex differences (Extended Data Figure 2)
% Load data
load('combined_preVpost_8020_nData.mat','lsdA_glm80_comb', ...
    'lsdA_glm80_comb_m','lsdA_glm80_comb_f',...
    'salA_glm80_comb','salA_glm80_comb_m','salA_glm80_comb_f')
figure
violinScatter({lsdA_glm80_comb,lsdA_glm80_comb_m,lsdA_glm80_comb_f,...
    salA_glm80_comb,salA_glm80_comb_m,salA_glm80_comb_f},'colors', ...
    {'r';'r';'r';'b';'b';'b'})
set(gca,'xticklabel',{'LSD','LSD M','LSD F','SAL','SAL M','SAL F'})
p = [];
p(1) = permutationTest(lsdA_glm80_comb,lsdA_glm80_comb_m,'+');
p(2) = permutationTest(lsdA_glm80_comb,lsdA_glm80_comb_f,'+');
p(3) = permutationTest(lsdA_glm80_comb_m,lsdA_glm80_comb_f,'+');
p(4) = permutationTest(salA_glm80_comb,salA_glm80_comb_m,'-');
p(5) = permutationTest(salA_glm80_comb,salA_glm80_comb_f,'-');
p(6) = permutationTest(salA_glm80_comb_m,salA_glm80_comb_f,'-');
sigstar({[1,2];[1,3];[2,3];[4,5];[4,6];[5,6]},p)
box off
ylim([0.4 1])
ylabel('AUC')
%% Velocity differences (Extended Data Figure 3)
load('H:\LSD+stim_persist\publicationData\movementData.mat')
figure
subplot(2,3,1)
violinScatter({basePre,basePost,base24Post,base48Post,base96Post}, ...
    'labels',{'pre','post','24 hr','48 hr','96 hr'})
hold on
plot([mean(basePre,'all'),mean(basePost,'all'),mean(base24Post,'all'), ...
    mean(base48Post,'all'), ...
    mean(base96Post,'all')],'-o','Color',[0.8 0.8 0.8])
ylabel('speed (pixel/min)')
box off
title('CTRL velocity')
ylim([0 30])

subplot(2,3,2)
violinScatter({salPre,salPost,sal24Post,sal48Post,sal72Post,sal144Post, ...
    sal168Post},'labels', ...
    {'pre','post','24 hr','48 hr','72 hr','144 hr', '168 hr'},'colors', ...
    {'b';'b';'b';'b';'b';'b';'b'})
hold on
plot([mean(salPre,'all'),mean(salPost,'all','omitnan'),mean(sal24Post,'all'), ...
    mean(sal48Post,'all'),mean(sal72Post,'all'),mean(sal144Post,'all'), ...
    mean(sal168Post,'all')],'-ok')
ylabel('speed (pixel/min)')
box off
title('SAL velocity')
ylim([0 30])

subplot(2,3,3)
violinScatter({lsdPre,lsdPost,lsd24Post,lsd48Post,lsd72Post,lsd144Post, ...
    lsd168Post},'labels', ...
    {'pre','post','24 hr','48 hr','72 hr','144 hr', '168 hr'},'colors', ...
    {'r';'r';'r';'r';'r';'r';'r'})
hold on
plot([mean(lsdPre,'all'),mean(lsdPost,'all','omitnan'),mean(lsd24Post,'all'), ...
    mean(lsd48Post,'all'),mean(lsd72Post,'all'),mean(lsd144Post,'all'), ...
    mean(lsd168Post,'all')],'-ok')
ylabel('speed (pixel/min)')
box off
title('LSD velocity')
ylim([0 30])

subplot(2,3,4)
violinScatter({aBase,aBase24,aBase48,aBase96}, ...
    'colors',{'k';'k';'k';'k'},'labels',...
    {'pre vs. post','24 hr','48 hr','96 hr'})
hold on
yline(mean(aBasePerm)+std(aBasePerm),'--')
yline(mean(aBasePerm)-std(aBasePerm),'--')
plot([mean(aBase),mean(aBase24),mean(aBase48),mean(aBase96)],'-o', ...
    'color',[0.8 0.8 0.8])
ylim([0.2 0.8])
box off

subplot(2,3,5)
violinScatter({aSAL,aSAL24,aSAL48,aSAL72,aSAL144,aSAL168}, ...
    'colors',{'b';'b';'b';'b';'b';'b'},'labels', ...
    {'pre vs. post','24 hr','48 hr','72 hr','144 hr','196 hr'})
hold on
yline(mean(aSALPerm)+std(aSALPerm),'--')
yline(mean(aSALPerm)-std(aSALPerm),'--')
plot([mean(aSAL),mean(aSAL24),mean(aSAL48),mean(aSAL72),mean(aSAL144), ...
    mean(aSAL168)],'-ok')
box off
ylim([0.2 0.8])
title('velocity classifiers')

subplot(2,3,6)
violinScatter({aLSD,aLSD24,aLSD48,aLSD72,aLSD144,aLSD168}, ...
    'colors',{'r';'r';'r';'r';'r';'r'},'labels', ...
    {'pre vs. post','24 hr','48 hr','72 hr','144 hr','196 hr'})
hold on
yline(mean(aLSDPerm)+std(aLSDPerm),'--')
yline(mean(aLSDPerm)-std(aLSDPerm),'--')
plot([mean(aLSD),mean(aLSD24),mean(aLSD48),mean(aLSD72),mean(aLSD144), ...
    mean(aLSD168)],'-ok')
box off
ylim([0.2 0.8])
%% Cohort comparison (Extended Data Figure 4)
mn = 0.3; mx = 1;
figure
% Pre vs. post
subplot(2,4,1)
violinScatter({lsdA_glm80_1hr,lsdA_glm80_1hr_perm,salA_glm80_1hr, ...
    salA_glm80_1hr_perm},...
    'colors',{'r','k','b','k'}','labels',...
    {'LSD','LSD perm','SAL','SAL perm'},'xvals',[1,1.5,2.25,2.75])
title('1 hr: pre vs. post')
p = [];
p(1) = permutationTest(lsdA_glm80_1hr,lsdA_glm80_1hr_perm,'+');
p(2) = permutationTest(lsdA_glm80_1hr,salA_glm80_1hr,'+');
p(3) = permutationTest(salA_glm80_1hr,salA_glm80_1hr_perm,'+');
sigstar({[1,1.5],[1,2.25],[2.25,2.75]},p);
ylabel('AUC')
box off
ylim([mn mx])

subplot(2,4,2)
violinScatter({lsdA_glm80_2hr,lsdA_glm80_2hr_perm,salA_glm80_2hr, ...
    salA_glm80_2hr_perm},...
    'colors',{'r','k','b','k'}','labels',...
    {'LSD','LSD perm','SAL','SAL perm'},'xvals',[1,1.5,2.25,2.75])
title('2 hr: pre vs. post')
p = [];
p(1) = permutationTest(lsdA_glm80_2hr,lsdA_glm80_2hr_perm,'+');
p(2) = permutationTest(lsdA_glm80_2hr,salA_glm80_2hr,'+');
p(3) = permutationTest(salA_glm80_2hr,salA_glm80_2hr_perm,'+');
sigstar({[1,1.5],[1,2.25],[2.25,2.75]},p);
ylabel('AUC')
box off
ylim([mn mx])

% Pre vs. post wash
subplot(2,4,5)
boxplot(lsdA_glm80_1hr_wash','colors','r','notch',1,'widths',0.25, ...
    'symbol','or')
hold on
boxplot(salA_glm80_1hr_wash','colors','b','notch',1,'widths',0.25, ...
    'positions',1.25:1:5.25,'symbol','ob')
p = NaN(1,5);
for ii = 1:5
    p(ii) = permutationTest(lsdA_glm80_1hr_wash(ii,:), ...
        salA_glm80_1hr_wash(ii,:),'+');
end
sigstar({[1,1.25],[2,2.25],[3,3.25],[4,4.25],[5,5.25]},p)
ylim([mn mx])
set(gca,'xticklabel',{'24 hr','48 hr','72 hr','144 hr','168 hr'})
yline(0.6662,'--k')
ylabel('AUC')
title('1 hr: washout')
box off

subplot(2,4,6)
boxplot(lsdA_glm80_2hr_wash','colors','r','notch',1,'widths',0.25,'symbol','or')
hold on
boxplot(salA_glm80_2hr_wash','colors','b','notch',1,'widths',0.25,...
    'positions',1.25:1:2.25,'symbol','ob')
p = [];
p(1) = permutationTest(lsdA_glm80_2hr_wash(1,:),salA_glm80_2hr_wash(1,:),'+');
p(2) = permutationTest(lsdA_glm80_2hr_wash(2,:),salA_glm80_2hr_wash(2,:),'+');
sigstar({[1,1.25],[2,2.25]},p)
ylim([mn mx])
set(gca,'xticklabel',{'24 hr','48 hr'})
yline(0.6662,'--k')
ylabel('AUC')
title('2 hr: washout')
box off

% Cross applied LSD↔SAL
subplot(2,4,3)
violinScatter({lsdA_glm80_1hr,lsdA_glm80_1hr_salTrain, ...
    salA_glm80_1hr_lsdTrain,salA_glm80_1hr},'colors',{'r';'c';'b';'m'}, ...
    'labels',{'LSD\rightarrowLSD','SAL\rightarrowLSD', ...
    'LSD\rightarrowSAL','SAL\rightarrowSAL'})
p = [];
p(1) = permutationTest(lsdA_glm80_1hr,lsdA_glm80_1hr_salTrain,'+');
p(2) = permutationTest(salA_glm80_1hr,lsdA_glm80_1hr_salTrain,'+');
p(3) = permutationTest(salA_glm80_1hr,salA_glm80_1hr_lsdTrain,'+');
p(4) = permutationTest(lsdA_glm80_1hr,salA_glm80_1hr_lsdTrain,'+');
sigstar({[1,2],[2,4],[3,4],[1,3]},p)
title('1 hr: cross-applied')
ylim([mn mx])
ylabel('AUC')
box off

subplot(2,4,4)
violinScatter({lsdA_glm80_2hr,lsdA_glm80_2hr_salTrain,salA_glm80_2hr,...
    salA_glm80_2hr_lsdTrain},'colors',{'r';'c';'b';'m'},'labels',...
    {'LSD\rightarrowLSD','SAL\rightarrowLSD','LSD\rightarrowSAL', ...
    'SAL\rightarrowSAL'})
p = [];
p(1) = permutationTest(lsdA_glm80_2hr,lsdA_glm80_2hr_salTrain,'+');
p(2) = permutationTest(salA_glm80_2hr,lsdA_glm80_2hr_salTrain,'+');
p(3) = permutationTest(salA_glm80_2hr,salA_glm80_2hr_lsdTrain,'+');
p(4) = permutationTest(lsdA_glm80_2hr,salA_glm80_2hr_lsdTrain,'+');
sigstar({[1,2],[2,4],[3,4],[1,3]},p)
ylim([mn mx])
ylabel('AUC')
box off
title('2 hr: cross-applied')

% Stim vs. post
subplot(2,4,7)
hold on
violinScatter({lsdA_glm80_1hr_stim,lsdA_glm80_1hr,...
    lsdA_glm80_1hr_stim_post,salA_glm80_1hr_stim,salA_glm80_1hr,...
    salA_glm80_1hr_stim_post},'colors',{[0.6479, 0.2000, 0.3714]; ...
    [1 0 0];[1.0000, 0.7414, 0.8965];[0.0860, 0.1908, 0.6159];[0 0 1]; ...
    [0.6028, 0.7485, 1.0000]})
ylim([mn mx])
ylabel('AUC')
box off
title('1 hr: stim vs. post')
set(gca,'xticklabel',{'S->S','P->P','S->P','S->S','P->P','S->P'})

p = NaN(1,6);
p(1) = permutationTest(lsdA_glm80_1hr_stim,lsdA_glm80_1hr,'+');
p(2) = permutationTest(lsdA_glm80_1hr_stim,lsdA_glm80_1hr_stim_post,'+');
p(3) = permutationTest(lsdA_glm80_1hr,lsdA_glm80_1hr_stim_post,'+');

p(4) = permutationTest(salA_glm80_1hr_stim,salA_glm80_1hr,'+');
p(5) = permutationTest(salA_glm80_1hr_stim,salA_glm80_1hr_stim_post,'+');
p(6) = permutationTest(salA_glm80_1hr,salA_glm80_1hr_stim_post,'+'); %#ok<NASGU>

subplot(2,4,8)
hold on
violinScatter({lsdA_glm80_2hr_stim,lsdA_glm80_2hr,...
    lsdA_glm80_2hr_stim_post,salA_glm80_2hr_stim,salA_glm80_2hr,...
    salA_glm80_2hr_stim_post},'colors',{[0.6479, 0.2000, 0.3714]; ...
    [1 0 0];[1.0000, 0.7414, 0.8965];[0.0860, 0.1908, 0.6159];[0 0 1]; ...
    [0.6028, 0.7485, 1.0000]})
ylim([mn mx])
ylabel('AUC')
box off
title('2 hr: stim vs. post')
set(gca,'xticklabel',{'S->S','P->P','S->P','S->S','P->P','S->P'})

p = NaN(1,6);
p(1) = permutationTest(lsdA_glm80_2hr_stim,lsdA_glm80_2hr,'+');
p(2) = permutationTest(lsdA_glm80_2hr_stim,lsdA_glm80_2hr_stim_post,'+');
p(3) = permutationTest(lsdA_glm80_2hr,lsdA_glm80_2hr_stim_post,'+');

p(4) = permutationTest(salA_glm80_2hr_stim,salA_glm80_2hr,'+');
p(5) = permutationTest(salA_glm80_2hr_stim,salA_glm80_2hr_stim_post,'+');
p(6) = permutationTest(salA_glm80_2hr,salA_glm80_2hr_stim_post,'+');