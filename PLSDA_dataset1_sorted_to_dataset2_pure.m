%% Cross-dataset PLS-DA
% Train: Dataset 1 positive-sorted B/T cells
% Test:  Dataset 2 pure/untouched B/T cells
%
% This adapter prepares the two datasets and then runs
% PLSDA_sorted_to_untouched_corrected.m, which performs:
%   - Dataset-1 batch-aware component selection
%   - training-only preprocessing
%   - one locked external test on Dataset 2
%   - accuracy, balanced accuracy, Q2, confusion and score plots
%   - VIP and ID-filtered VIP analysis
%   - exact Dataset-1 batch-blocked permutation testing

clear;
clc;

%% Paths
Loc1 = "D:\CITE\MACS cells\MACS9\2026-02-16_MACS9_320-1000_neg_S100_150um_3ul_2bar";
FN1 = "NewProcessing_HighNeg_550.csv";
meta1Path = "Metadata_newprocessing.xlsx";

Loc2 = "D:\CITE\MACS cells\MACS8";
FN2 = "HighNeg_IntensityMatrix_NewProcessing_550.csv";
meta2Path = "HighNeg_Metadata.xlsx";

root_save_dir = fullfile(Loc1, ...
    "PLSDA_Dataset1Sorted_to_Dataset2Pure_" + ...
    string(datetime('now','Format','yyyyMMdd')));

if ~exist(root_save_dir,'dir')
    mkdir(root_save_dir);
end

%% Metadata settings -- confirm these names/values
dataset1ClassColumn = "Cell";
dataset1BatchColumn = "Batch";
dataset1ConditionColumn = "Condition";
dataset1PositiveSortedValue = "Positive Sorted";

dataset2ClassColumn = "Cell";

% Dataset 2 identifies its pure samples through metadata2.Sorting.
% In the supplied script, pure samples are called B and T in that column.
dataset2PureColumn = "Sorting";
dataset2PureValues = ["B","T"];

% Feature matching tolerance. Use a value appropriate for the instrument.
mzToleranceDa = 0.01;

%% Load datasets
data1 = readtable(fullfile(Loc1,FN1));
metadata1 = readtable(fullfile(Loc1,meta1Path));

data2 = readtable(fullfile(Loc2,FN2));
metadata2 = readtable(fullfile(Loc2,meta2Path));

MZlist1 = double(table2array(data1(1,2:end)));
ROIlist1 = string(data1{2:end,1});
rawINT1 = double(table2array(data1(2:end,2:end)));

MZlist2 = double(table2array(data2(1,2:end)));
ROIlist2 = string(data2{2:end,1});
rawINT2 = double(table2array(data2(2:end,2:end)));

% LEAKAGE TEST: shuffle Dataset 2 intensities within each m/z feature
rng(1);
for j = 1:size(rawINT2,2)
    rawINT2(:,j) = rawINT2(randperm(size(rawINT2,1)), j);
end

if size(rawINT1,1) ~= height(metadata1)
    error('Dataset 1 intensity rows and metadata rows do not match.');
end

if size(rawINT2,1) ~= height(metadata2)
    error('Dataset 2 intensity rows and metadata rows do not match.');
end

%% Validate required metadata columns
required1 = [dataset1ClassColumn,dataset1BatchColumn, ...
    dataset1ConditionColumn];
required2 = [dataset2ClassColumn,dataset2PureColumn];

names1 = string(metadata1.Properties.VariableNames);
names2 = string(metadata2.Properties.VariableNames);

missing1 = required1(~ismember(required1,names1));
missing2 = required2(~ismember(required2,names2));

if ~isempty(missing1)
    error('Missing Dataset 1 metadata columns: %s', ...
        strjoin(missing1,', '));
end

if ~isempty(missing2)
    error('Missing Dataset 2 metadata columns: %s', ...
        strjoin(missing2,', '));
end

%% Select intended populations before modelling
cellLabels1 = string(metadata1.(dataset1ClassColumn));
batchValues1 = string(metadata1.(dataset1BatchColumn));
conditionValues1 = string(metadata1.(dataset1ConditionColumn));

cellLabels2 = string(metadata2.(dataset2ClassColumn));
pureGroupValues2 = string(metadata2.(dataset2PureColumn));

trainMask1 = conditionValues1 == dataset1PositiveSortedValue & ...
    ismember(cellLabels1,["B","T"]);

testMask2 = ismember(pureGroupValues2,dataset2PureValues) & ...
    ismember(cellLabels2,["B","T"]);

if ~any(trainMask1)
    error(['No Dataset 1 positive-sorted B/T rows were found. ' ...
        'Check dataset1ConditionColumn and dataset1PositiveSortedValue.']);
end

if ~any(testMask2)
    error(['No Dataset 2 pure B/T rows were found. ' ...
        'Check dataset2PureColumn and dataset2PureValues.']);
end

fprintf('\nSelected populations\n');
fprintf('Dataset 1 positive-sorted training rows: %d\n',sum(trainMask1));
fprintf('Dataset 2 pure test rows: %d\n',sum(testMask2));
disp('Dataset 1 training conditions:');
disp(unique(conditionValues1(trainMask1)));
disp('Dataset 2 pure-group values:');
disp(unique(pureGroupValues2(testMask2)));

%% One-to-one m/z alignment
% Candidate matches are sorted by mass difference and accepted greedily,
% ensuring that a feature in either dataset is used at most once.
[candidateI,candidateJ] = find( ...
    abs(MZlist1(:)-MZlist2(:)') <= mzToleranceDa);

if isempty(candidateI)
    error('No shared m/z features were found within %.4f Da.',mzToleranceDa);
end

candidateDifference = abs( ...
    MZlist1(candidateI)-MZlist2(candidateJ));
[~,candidateOrder] = sort(candidateDifference,'ascend');

used1 = false(numel(MZlist1),1);
used2 = false(numel(MZlist2),1);
idx1Shared = zeros(0,1);
idx2Shared = zeros(0,1);

for k = candidateOrder(:)'
    i = candidateI(k);
    j = candidateJ(k);

    if ~used1(i) && ~used2(j)
        idx1Shared(end+1,1) = i; %#ok<SAGROW>
        idx2Shared(end+1,1) = j; %#ok<SAGROW>
        used1(i) = true;
        used2(j) = true;
    end
end

% Put aligned features into ascending Dataset-1 m/z order.
[~,sharedOrder] = sort(MZlist1(idx1Shared),'ascend');
idx1Shared = idx1Shared(sharedOrder);
idx2Shared = idx2Shared(sharedOrder);

MZlist = MZlist1(idx1Shared);
dataset2MatchedMZ = MZlist2(idx2Shared);
massDifferenceDa = abs(MZlist-dataset2MatchedMZ);

fprintf('Dataset 1 features: %d\n',numel(MZlist1));
fprintf('Dataset 2 features: %d\n',numel(MZlist2));
fprintf('One-to-one aligned features: %d\n',numel(MZlist));
fprintf('Maximum accepted difference: %.6f Da\n',max(massDifferenceDa));

if numel(MZlist) < 10
    warning('Only %d shared features were found.',numel(MZlist));
end

alignmentTable = table( ...
    MZlist(:),dataset2MatchedMZ(:),massDifferenceDa(:), ...
    'VariableNames',{'Dataset1MZ','Dataset2MZ','DifferenceDa'});
writetable(alignmentTable,fullfile(root_save_dir,'MZ_alignment.csv'));

%% Build the standardized workspace expected by the validated script
% Only the intended training and external-test populations are included.
X1selected = rawINT1(trainMask1,idx1Shared);
X2selected = rawINT2(testMask2,idx2Shared);


rawINT = [X1selected;X2selected];
ROIlist = [ROIlist1(trainMask1);ROIlist2(testMask2)];

selectedClassLabels = [ ...
    cellLabels1(trainMask1); ...
    cellLabels2(testMask2)];

% Force a stable and explicit B=1, T=2 mapping.
ClassName1 = ["B";"T"];
Class1 = zeros(numel(selectedClassLabels),1);
Class1(selectedClassLabels == "B") = 1;
Class1(selectedClassLabels == "T") = 2;

if any(Class1 == 0)
    error('An unexpected class label remained after B/T filtering.');
end

nTrainSelected = sum(trainMask1);
nTestSelected = sum(testMask2);

combinedCondition = [ ...
    repmat("Positive Sorted",nTrainSelected,1); ...
    repmat("Naive",nTestSelected,1)];

% Dataset-1 batches drive component selection and blocked permutations.
% Dataset-2 batch values are retained for reporting when available.
if ismember(dataset1BatchColumn,names2)
    dataset2BatchValues = string( ...
        metadata2.(dataset1BatchColumn));
    dataset2BatchValues = dataset2BatchValues(testMask2);
else
    dataset2BatchValues = repmat("Dataset2",nTestSelected,1);
end

combinedBatch = [ ...
    batchValues1(trainMask1); ...
    dataset2BatchValues];

metadata = table(combinedCondition,combinedBatch, ...
    'VariableNames',{'Condition','Batch'});

%% Locate the corrected validated workflow
analysisScript = fullfile(fileparts(mfilename('fullpath')), ...
    'PLSDA_sorted_to_untouched_corrected.m');

if ~isfile(analysisScript)
    error('Required analysis script was not found: %s',analysisScript);
end

%% Model 1: Dataset 1 positive-sorted -> Dataset 2 pure
save_dir = fullfile(root_save_dir, ...
    'Model1_Dataset1_sortplus_to_Dataset2_pure');
if ~exist(save_dir,'dir'); mkdir(save_dir); end

trainingPlotSuffix = "_{sort+}";
testPlotSuffix = "_{pure}";
plotFontSize = 18;
modelPlotTitle = "Dataset 1 sort+ -> Dataset 2 pure";

run(analysisScript);

%% ============================================================
% Model 2: all Dataset 1 B/T samples -> all Dataset 2 B/T samples
% =============================================================
allTrainMask1 = ismember(cellLabels1,["B","T"]);
allTestMask2 = ismember(cellLabels2,["B","T"]);

if ~any(allTrainMask1) || ~any(allTestMask2)
    error('Model 2 requires B/T observations in both datasets.');
end

rawINT = [ ...
    rawINT1(allTrainMask1,idx1Shared); ...
    rawINT2(allTestMask2,idx2Shared)];

ROIlist = [ROIlist1(allTrainMask1);ROIlist2(allTestMask2)];
allModelLabels = [cellLabels1(allTrainMask1);cellLabels2(allTestMask2)];

ClassName1 = ["B";"T"];
Class1 = zeros(numel(allModelLabels),1);
Class1(allModelLabels == "B") = 1;
Class1(allModelLabels == "T") = 2;

nAllTrain1 = sum(allTrainMask1);
nAllTest2 = sum(allTestMask2);

combinedCondition = [ ...
    repmat("Positive Sorted",nAllTrain1,1); ...
    repmat("Naive",nAllTest2,1)];

if ismember(dataset1BatchColumn,names2)
    allDataset2Batches = string(metadata2.(dataset1BatchColumn));
    allDataset2Batches = allDataset2Batches(allTestMask2);
else
    allDataset2Batches = repmat("Dataset2",nAllTest2,1);
end

combinedBatch = [ ...
    batchValues1(allTrainMask1); ...
    allDataset2Batches];

metadata = table(combinedCondition,combinedBatch, ...
    'VariableNames',{'Condition','Batch'});

save_dir = fullfile(root_save_dir, ...
    'Model2_AllDataset1_to_AllDataset2');
if ~exist(save_dir,'dir'); mkdir(save_dir); end

trainingPlotSuffix = "_{D1}";
testPlotSuffix = "_{D2}";
plotFontSize = 18;
modelPlotTitle = "All Dataset 1 B/T -> all Dataset 2 B/T";

fprintf('\n====================================================\n');
fprintf('MODEL 2: all Dataset 1 B/T -> all Dataset 2 B/T\n');
fprintf('Dataset 1 training observations: %d\n',nAllTrain1);
fprintf('Dataset 2 test observations: %d\n',nAllTest2);

run(analysisScript);
