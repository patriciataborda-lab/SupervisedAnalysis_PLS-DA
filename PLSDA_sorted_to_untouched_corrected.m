%% PLS-DA: train on positive-sorted cells, predict pure cells
%
% Primary analysis:
%   1. Select component count by leave-one-batch-out validation using
%      positive-sorted cells only.
%   2. Fit one final model using all positive-sorted cells.
%   3. Evaluate the final model once using all pure ("Naive") cells.
%   4. Calculate external accuracy, balanced accuracy, Q2, and confusion matrix.
%   5. Perform an exact batch-blocked permutation test by swapping B/T labels
%      within each positive-sorted analytical batch.
%
% Expected variables already in the workspace:
%   rawINT, Class1, ClassName1, ROIlist, metadata, save_dir
%
% Required metadata columns:
%   metadata.Condition
%   metadata.Batch

%% User settings
maximumComponents = 10;
trainingCondition = "Positive Sorted";
testCondition = "Naive";  % Change to "Pure" if that is the metadata label
topVIPFraction = 0.05;     % 0.05 = top 5%; use 0.01 for top 1%

% TeX-formatted plot suffixes; adapter scripts may define these first.
if ~exist('trainingPlotSuffix','var'); trainingPlotSuffix = "_{sort+}"; end
if ~exist('testPlotSuffix','var'); testPlotSuffix = "_{pure}"; end
if ~exist('plotFontSize','var'); plotFontSize = 18; end
if ~exist('modelPlotTitle','var')
    modelPlotTitle = "Positive-sorted training -> pure external test";
end

% Optional VIP annotation settings
% Set this to the CSV/XLSX file containing an m/z column and an ID column.
vipAnnotationFile = ...
    "D:\CITE\MACS cells\MACS9\2026-02-16_MACS9_320-1000_neg_S100_150um_3ul_2bar\HighNeg_CrossValidation_20260707\top_VIP_features_comparison_withID_noIsotopes.csv";
vipAnnotationMZColumn = "Feature";
vipAnnotationIDColumn = "ID";
vipAnnotationTolerancePPM = 5;

X = rawINT;
originalLabels = double(Class1(:));
classNamesOriginal = string(ClassName1(:));
sampleNames = string(ROIlist(:));
conditionLabels = string(metadata.Condition(:));
batchLabels = string(metadata.Batch(:));

%% Validate inputs
nObservations = size(X,1);

if numel(originalLabels) ~= nObservations || ...
        numel(sampleNames) ~= nObservations || ...
        numel(conditionLabels) ~= nObservations || ...
        numel(batchLabels) ~= nObservations
    error('X, Class1, ROIlist, metadata.Condition, and metadata.Batch must have matching row counts.');
end

if any(ismissing(conditionLabels)) || any(ismissing(batchLabels))
    error('Condition and batch labels cannot contain missing values.');
end

trainMask = conditionLabels == trainingCondition;
testMask = conditionLabels == testCondition;

if ~any(trainMask)
    error('No observations have condition "%s".', trainingCondition);
end

if ~any(testMask)
    error('No observations have condition "%s".', testCondition);
end

%% Map class labels to consecutive indices 1,...,K
classIDs = unique(originalLabels, 'sorted');
nClasses = numel(classIDs);
classIndex = zeros(nObservations,1);

for c = 1:nClasses
    classIndex(originalLabels == classIDs(c)) = c;
end

if all(classIDs >= 1) && all(classIDs <= numel(classNamesOriginal)) && ...
        all(mod(classIDs,1) == 0)
    classNames = classNamesOriginal(classIDs);
else
    classNames = "Class_" + string(classIDs);
end

Ydummy = zeros(nObservations,nClasses);
for c = 1:nClasses
    Ydummy(:,c) = classIndex == c;
end

XtrainRaw = X(trainMask,:);
Ytrain = Ydummy(trainMask,:);
YtrainIndex = classIndex(trainMask);
trainBatches = batchLabels(trainMask);

XtestRaw = X(testMask,:);

Ytest = Ydummy(testMask,:);
YtestIndex = classIndex(testMask);
testSamples = sampleNames(testMask);
testBatches = batchLabels(testMask);

if numel(unique(YtrainIndex)) ~= nClasses
    error('The training population does not contain every class.');
end

if numel(unique(YtestIndex)) ~= nClasses
    warning('The pure test population does not contain every class.');
end

sortedBatchNames = unique(trainBatches, 'stable');
nSortedBatches = numel(sortedBatchNames);

if nSortedBatches < 2
    error('At least two training batches are required.');
end

fprintf('\n====================================================\n');
fprintf('PLS-DA: training model -> external test\n');
fprintf('====================================================\n');
fprintf('Training observations: %d\n', size(XtrainRaw,1));
fprintf('External test observations: %d\n', size(XtestRaw,1));
fprintf('Training batches: %d\n', nSortedBatches);

%% Select component count using sorted cells only
[optimalComponents, cvAccuracy, cvBalancedAccuracy, cvQ2, cvPredictions] = ...
    selectComponentsByBatch( ...
        XtrainRaw, Ytrain, YtrainIndex, trainBatches, maximumComponents);

fprintf('\nSelected components: %d\n', optimalComponents);
fprintf('Training batch-CV accuracy: %.4f\n', ...
    cvAccuracy(optimalComponents));
fprintf('Training batch-CV balanced accuracy: %.4f\n', ...
    cvBalancedAccuracy(optimalComponents));
fprintf('Training batch-CV Q2: %.4f\n', cvQ2(optimalComponents));

%% Fit final model using every positive-sorted observation
scale90Final = prctile(XtrainRaw,90,1);
scale90Final(~isfinite(scale90Final) | scale90Final == 0) = 1;

Xtrain = XtrainRaw ./ scale90Final;
Xtest = XtestRaw ./ scale90Final;

[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = ...
    plsregress(Xtrain, Ytrain, optimalComponents);


% Sensitivity analysis: PLS-DA component count
% Run for Model 2, after Xtrain and Xtest have been scaled.
disp('>>> STARTING COMPONENT COMPARISON <<<');
componentNumbers = 1:min(7, size(Xtrain,1)-1);

nComponents = numel(componentNumbers);
accuracyByComponent = nan(nComponents,1);
balancedAccuracyByComponent = nan(nComponents,1);

% Class labels: B = 1, T = 2
trueClass = Ytest;

% If Ytest is a two-column dummy matrix:
if size(Ytest,2) == 2
    [~,trueClass] = max(Ytest,[],2);
end

for i = 1:nComponents

    nComp = componentNumbers(i);

    [~,~,~,~,BETA] = plsregress( ...
        Xtrain,Ytrain,nComp);

    % Predict the independent dataset
    predictedY = [ones(size(Xtest,1),1),Xtest] * BETA;

    % Select the class with the highest predicted response
    [~,predictedClass] = max(predictedY,[],2);

    % Overall accuracy
    accuracyByComponent(i) = ...
        mean(predictedClass == trueClass);

    % Balanced accuracy: mean of B and T recall
    recallB = mean(predictedClass(trueClass==1) == 1);
    recallT = mean(predictedClass(trueClass==2) == 2);

    balancedAccuracyByComponent(i) = ...
        mean([recallB,recallT]);

end

results = table( ...
    componentNumbers(:), ...
    accuracyByComponent, ...
    balancedAccuracyByComponent, ...
    'VariableNames', ...
    {'Components','Accuracy','BalancedAccuracy'});

% Save the component-comparison table
writetable(results, ...
    fullfile(pwd, 'Model2_ComponentComparison.csv'));

figure;
plot(results.Components,results.Accuracy*100, ...
    '-o','LineWidth',1.5);
hold on;
plot(results.Components,results.BalancedAccuracy*100, ...
    '-s','LineWidth',1.5);
xlabel('Number of PLS-DA components');
ylabel('Independent test accuracy (%)');
legend('Overall accuracy','Balanced accuracy', ...
    'Location','best');
grid on;
exportgraphics(gcf, ...
    fullfile(pwd, 'Model2_ComponentComparison.png'), ...
    'Resolution', 300);
disp('>>> COMPONENT COMPARISON FINISHED <<<');
disp('Checking component comparison results:');

if exist('results', 'var')
    disp(results);
else
    disp('ERROR: The variable results does not exist.');
end


%% Plot external PLS-DA prediction scores
% Uses the CV-selected model, not the component comparison models.

% Fit the selected model on the full training dataset
[~,~,~,~,betaPlot] = plsregress( ...
    Xtrain, Ytrain, optimalComponents);

% Predict dataset 2
YpredPlot = [ones(size(Xtest,1),1), Xtest] * betaPlot;

% The first response column represents B cells (B = 1)
scoreB = YpredPlot(:,1);

% Get actual class labels
if size(Ytest,2) == 2
    [~,trueClassPlot] = max(Ytest,[],2);
else
    trueClassPlot = Ytest(:);
end

% Predicted class
[~,predictedClassPlot] = max(YpredPlot,[],2);

% Identify incorrectly classified observations
incorrectPlot = predictedClassPlot ~= trueClassPlot;

% Set up figure
fig = figure('Color','w', ...
    'Position',[100 100 850 550]);

hold on;

% Make points reproducibly spaced (no random jitter)
idxB = find(trueClassPlot == 1);
idxT = find(trueClassPlot == 2);

xB = 1 + linspace(-0.16,0.16,numel(idxB))';
xT = 2 + linspace(-0.16,0.16,numel(idxT))';

% Plot true B cells in blue
scatter(xB,scoreB(idxB),55, ...
    [0.12 0.42 0.72], ...
    'filled', ...
    'MarkerFaceAlpha',0.75);

% Plot true T cells in orange
scatter(xT,scoreB(idxT),55, ...
    [0.90 0.47 0.18], ...
    'filled', ...
    'MarkerFaceAlpha',0.75);

% Highlight misclassified observations
xAll = zeros(size(trueClassPlot));
xAll(idxB) = xB;
xAll(idxT) = xT;

scatter(xAll(incorrectPlot), ...
    scoreB(incorrectPlot), ...
    150,'k','o', ...
    'LineWidth',1.8);

% Classification boundary
yline(0.5,'k--', ...
    'Decision boundary', ...
    'LineWidth',1.3, ...
    'LabelHorizontalAlignment','left');

% Ideal class-response values
yline(1,':','Color',[0.5 0.5 0.5]);
yline(0,':','Color',[0.5 0.5 0.5]);

% Format axes
xlim([0.5 2.5]);
xticks([1 2]);
xticklabels({'B cells','T cells'});

ylabel('Predicted B-cell response (PLS-DA)');
xlabel('Actual cell type');

title(sprintf( ...
    'Independent dataset predictions (%d PLS components)', ...
    optimalComponents));

set(gca,'FontSize',12, ...
    'LineWidth',1, ...
    'Box','off');

grid on;
hold off;

% Save to MATLAB's current working folder
outputPNG = fullfile(pwd, ...
    'Model2_PLSDA_ExternalPredictionScores.png');

outputFIG = fullfile(pwd, ...
    'Model2_PLSDA_ExternalPredictionScores.fig');

exportgraphics(fig,outputPNG,'Resolution',300);
savefig(fig,outputFIG);

fprintf('Prediction score figure saved to:\n%s\n',outputPNG);
fprintf('Number of misclassified observations: %d\n', ...
    sum(incorrectPlot));



%% Predict pure cells once
YtestPredicted = [ones(size(Xtest,1),1), Xtest] * BETA;
[~, predictedTestClass] = max(YtestPredicted,[],2);

externalAccuracy = mean(predictedTestClass == YtestIndex);

externalConfusionMatrix = confusionmat( ...
    YtestIndex, predictedTestClass, 'Order',1:nClasses);

externalRecall = diag(externalConfusionMatrix) ./ ...
    sum(externalConfusionMatrix,2);
externalBalancedAccuracy = mean(externalRecall,'omitnan');

%% External predictive Q2
PRESS_external = sum((Ytest - YtestPredicted).^2,'all');

% Use the mean training response as the no-model prediction.
meanTrainingResponse = mean(Ytrain,1);
TSS_external = sum((Ytest - meanTrainingResponse).^2,'all');

if TSS_external > 0
    externalQ2 = 1 - PRESS_external/TSS_external;
else
    externalQ2 = NaN;
end

fprintf('\nExternal test performance\n');
fprintf('Accuracy: %.4f\n', externalAccuracy);
fprintf('Balanced accuracy: %.4f\n', externalBalancedAccuracy);
fprintf('Predictive Q2: %.4f\n', externalQ2);
disp('External confusion matrix:');
disp(array2table(externalConfusionMatrix, ...
    'VariableNames',"Predicted_" + classNames, ...
    'RowNames',"True_" + classNames));

%% Export external predictions
trueClassText = classNames(YtestIndex);
predictedClassText = classNames(predictedTestClass);

predictionTable = table( ...
    testSamples, testBatches, trueClassText, predictedClassText, ...
    predictedTestClass == YtestIndex, ...
    'VariableNames', ...
    {'Sample','Batch','TrueClass','PredictedClass','Correct'});

for c = 1:nClasses
    scoreName = matlab.lang.makeValidName("Score_" + classNames(c));
    predictionTable.(scoreName) = YtestPredicted(:,c);
end

writetable(predictionTable, fullfile(save_dir, ...
    'PLSDA_Training_to_ExternalTest_predictions.csv'));

%% Export summary
summaryTable = table( ...
    size(XtrainRaw,1), size(XtestRaw,1), nSortedBatches, ...
    optimalComponents, cvAccuracy(optimalComponents), ...
    cvBalancedAccuracy(optimalComponents), cvQ2(optimalComponents), ...
    externalAccuracy, externalBalancedAccuracy, externalQ2, ...
    'VariableNames',{ ...
    'NumberTrainingObservations', ...
    'NumberExternalTestObservations', ...
    'NumberTrainingBatches', ...
    'SelectedComponents', ...
    'TrainingBatchCVAccuracy', ...
    'TrainingBatchCVBalancedAccuracy', ...
    'TrainingBatchCVQ2', ...
    'ExternalAccuracy', ...
    'ExternalBalancedAccuracy', ...
    'ExternalQ2'});

writetable(summaryTable, fullfile(save_dir, ...
    'PLSDA_Training_to_ExternalTest_summary.csv'));

disp(summaryTable);

%% Plot component selection
figComponents = figure('Color','w','Position',[100 100 800 550]);

yyaxis left
plot(1:numel(cvBalancedAccuracy), cvBalancedAccuracy, ...
    '-o','LineWidth',1.5,'DisplayName','Balanced accuracy');
ylabel('Balanced accuracy');
ylim([0 1.05]);

yyaxis right
plot(1:numel(cvQ2), cvQ2, ...
    '-s','LineWidth',1.5,'DisplayName','Q^2');
ylabel('Q^2');

xline(optimalComponents,'k--','Selected','LineWidth',1.2);
xlabel('Number of PLS components');
title(modelPlotTitle + ": training-only component selection");
grid on;

standardizeFigureFonts(figComponents,plotFontSize);

exportgraphics(figComponents, fullfile(save_dir, ...
    'PLSDA_SortedTrain_component_selection.png'),'Resolution',300);
saveas(figComponents, fullfile(save_dir, ...
    'PLSDA_SortedTrain_component_selection.fig'));

%% Plot external confusion matrix
trueCategorical = categorical(trueClassText,classNames);
predictedCategorical = categorical(predictedClassText,classNames);

figCM = figure('Color','w','Position',[100 100 750 600]);
cmChart = confusionchart( ...
    trueCategorical, predictedCategorical, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

cmChart.Title = sprintf( ...
    [char(modelPlotTitle) '\n' ...
    'Accuracy = %.2f | Balanced accuracy = %.2f | Q^2 = %.2f'], ...
    externalAccuracy, externalBalancedAccuracy, externalQ2);
cmChart.XLabel = 'Predicted cell type';
cmChart.YLabel = 'True cell type';
cmChart.FontName = 'Arial';
cmChart.FontSize = plotFontSize;

standardizeFigureFonts(figCM,plotFontSize);

exportgraphics(figCM, fullfile(save_dir, ...
    'PLSDA_Training_to_ExternalTest_confusion.png'),'Resolution',300);
saveas(figCM, fullfile(save_dir, ...
    'PLSDA_Training_to_ExternalTest_confusion.fig'));

%% ============================================================
% Loading plot for the selected predictive model
%
% The validated model may contain only one component. This plot therefore
% shows each feature's X loading on component 1. Large positive or negative
% values identify variables contributing strongly to the latent component.
% =============================================================
nFeatures = size(Xtrain,2);

if exist('variableNames','var') && numel(variableNames) == nFeatures
    loadingFeatureLabels = string(variableNames(:));
elseif exist('MZlist','var') && numel(MZlist) == nFeatures
    loadingFeatureLabels = string(MZlist(:));
else
    loadingFeatureLabels = "Feature_" + string((1:nFeatures)');
end

% Use numeric m/z values on the horizontal axis when available.
useNumericMZAxis = false;
if exist('MZlist','var') && numel(MZlist) == nFeatures && isnumeric(MZlist)
    loadingXAxis = double(MZlist(:));
    loadingXAxisLabel = 'm/z';
    useNumericMZAxis = true;
else
    loadingXAxis = (1:nFeatures)';
    loadingXAxisLabel = 'Feature index';
end

component1Loadings = XL(:,1);

% Label the largest 1% of absolute loadings, with a minimum of five.
nLoadingLabels = min(nFeatures,max(5,ceil(0.01*nFeatures)));
[~,absoluteLoadingOrder] = sort(abs(component1Loadings),'descend');
topLoadingRows = absoluteLoadingOrder(1:nLoadingLabels);

figLoading1 = figure('Color','w','Position',[100 100 1000 600]);
scatter(loadingXAxis,component1Loadings,18, ...
    [0.55 0.55 0.55],'filled');
hold on;
scatter(loadingXAxis(topLoadingRows),component1Loadings(topLoadingRows), ...
    45,[0.80 0.20 0.20],'filled');
yline(0,'k-');

for j = 1:numel(topLoadingRows)
    row = topLoadingRows(j);
    text(loadingXAxis(row),component1Loadings(row), ...
        "  " + loadingFeatureLabels(row), ...
        'FontSize',8,'Interpreter','none');
end

hold off;
xlabel(loadingXAxisLabel);
ylabel('X loading');
title(sprintf( ...
    'Selected PLS-DA model: component 1 loadings (%d component model)', ...
    optimalComponents));
grid on;
box on;

standardizeFigureFonts(figLoading1,plotFontSize);

exportgraphics(figLoading1, fullfile(save_dir, ...
    'PLSDA_SortedTrain_selected_model_component1_loadings.png'), ...
    'Resolution',300);
saveas(figLoading1, fullfile(save_dir, ...
    'PLSDA_SortedTrain_selected_model_component1_loadings.fig'));

loadingTable = table( ...
    loadingFeatureLabels,component1Loadings,abs(component1Loadings), ...
    'VariableNames',{'Feature','Loading_Component1','AbsoluteLoading'});

if useNumericMZAxis
    loadingTable.MZ = loadingXAxis;
    loadingTable = movevars(loadingTable,'MZ','After','Feature');
end

loadingTable = sortrows(loadingTable,'AbsoluteLoading','descend');
writetable(loadingTable,fullfile(save_dir, ...
    'PLSDA_SortedTrain_selected_model_loadings.csv'));

%% ============================================================
% Optional two-component descriptive score and loading plots
%
% If the validated model selected only one component, a separate
% two-component model is fitted solely for visualization. It is not used
% to calculate accuracy, Q2, predictions, or permutation p-values.
% =============================================================
maximumVisualComponents = min([ ...
    2,size(Xtrain,1)-1,size(Xtrain,2),rank(Xtrain)]);

if maximumVisualComponents >= 2
    [XL_visual,~,XS_visual,~,~,PCTVAR_visual,~,stats_visual] = ...
        plsregress(Xtrain,Ytrain,2);

    % plsregress centers X internally. Project untouched observations using
    % the training mean and the fitted PLS weights.
    meanXtrain = mean(Xtrain,1);
    untouchedScoresVisual = ...
        (Xtest-meanXtrain) * stats_visual.W(:,1:2);

    explainedXVisual = 100*PCTVAR_visual(1,1:2);

    % ---------------------------------------------------------
    % Two-component score plot
    % ---------------------------------------------------------
    baseColors = lines(nClasses);
    lightColors = baseColors + (1-baseColors)*0.55;

    figScoresVisual = figure('Color','w','Position',[100 100 900 650]);
    hold on;

    for c = 1:nClasses
        sortedRows = YtrainIndex == c;
        untouchedRows = YtestIndex == c;

        scatter(XS_visual(sortedRows,1),XS_visual(sortedRows,2), ...
            85,baseColors(c,:),'o','filled', ...
            'MarkerEdgeColor','none', ...
            'DisplayName',classNames(c) + trainingPlotSuffix);

        scatter(untouchedScoresVisual(untouchedRows,1), ...
            untouchedScoresVisual(untouchedRows,2), ...
            105,lightColors(c,:),'o','filled', ...
            'MarkerEdgeColor',baseColors(c,:), ...
            'LineWidth',1.2, ...
            'DisplayName',classNames(c) + testPlotSuffix);
    end

    % Mark errors made by the selected predictive model. Point colors still
    % represent the true classes; the black X indicates misclassification.
    misclassifiedTestRows = predictedTestClass ~= YtestIndex;
    if any(misclassifiedTestRows)
        scatter( ...
            untouchedScoresVisual(misclassifiedTestRows,1), ...
            untouchedScoresVisual(misclassifiedTestRows,2), ...
            190,'k','x','LineWidth',2.5, ...
            'DisplayName','Misclassified by selected model');

        misclassifiedRows = find(misclassifiedTestRows);
        for errorIndex = 1:numel(misclassifiedRows)
            row = misclassifiedRows(errorIndex);
            text(untouchedScoresVisual(row,1), ...
                untouchedScoresVisual(row,2), ...
                "  " + testSamples(row), ...
                'Color','k','FontWeight','bold', ...
                'FontSize',max(11,plotFontSize-2), ...
                'Interpreter','none');
        end
    end

    hold off;
    xlabel(sprintf('PLS 1 (%.2f%%)',explainedXVisual(1)));
    ylabel(sprintf('PLS 2 (%.2f%%)',explainedXVisual(2)));
    title(char(modelPlotTitle));
    legend('Location','best');
    grid on;
    box on;
    set(gca,'FontSize',plotFontSize,'LineWidth',1.2);

    standardizeFigureFonts(figScoresVisual,plotFontSize);

    exportgraphics(figScoresVisual,fullfile(save_dir, ...
        'PLSDA_Training_ExternalTest_2component_scores.png'), ...
        'Resolution',300);
    saveas(figScoresVisual,fullfile(save_dir, ...
        'PLSDA_Training_ExternalTest_2component_scores.fig'));

    % ---------------------------------------------------------
    % Two-component loading plot
    % ---------------------------------------------------------
    loadingMagnitudeVisual = sqrt(sum(XL_visual(:,1:2).^2,2));
    [~,loadingOrderVisual] = sort(loadingMagnitudeVisual,'descend');
    topLoadingRowsVisual = loadingOrderVisual(1:nLoadingLabels);

    figLoadingsVisual = figure('Color','w','Position',[100 100 800 700]);
    scatter(XL_visual(:,1),XL_visual(:,2),18, ...
        [0.60 0.60 0.60],'filled');
    hold on;
    scatter(XL_visual(topLoadingRowsVisual,1), ...
        XL_visual(topLoadingRowsVisual,2), ...
        50,[0.80 0.20 0.20],'filled');
    xline(0,'k-');
    yline(0,'k-');

    for j = 1:numel(topLoadingRowsVisual)
        row = topLoadingRowsVisual(j);
        text(XL_visual(row,1),XL_visual(row,2), ...
            "  " + loadingFeatureLabels(row), ...
            'FontSize',8,'Interpreter','none');
    end

    hold off;
    xlabel('X loading, component 1');
    ylabel('X loading, component 2');
    title({'PLS-DA X-loading plot', ...
        'Two-component model fitted for visualization only'});
    grid on;
    box on;

    standardizeFigureFonts(figLoadingsVisual,plotFontSize);

    exportgraphics(figLoadingsVisual,fullfile(save_dir, ...
        'PLSDA_SortedTrain_2component_loadings.png'), ...
        'Resolution',300);
    saveas(figLoadingsVisual,fullfile(save_dir, ...
        'PLSDA_SortedTrain_2component_loadings.fig'));

    visualLoadingTable = table( ...
        loadingFeatureLabels,XL_visual(:,1),XL_visual(:,2), ...
        loadingMagnitudeVisual, ...
        'VariableNames',{ ...
        'Feature','Loading_Component1','Loading_Component2', ...
        'LoadingMagnitude'});

    if useNumericMZAxis
        visualLoadingTable.MZ = loadingXAxis;
        visualLoadingTable = movevars( ...
            visualLoadingTable,'MZ','After','Feature');
    end

    visualLoadingTable = sortrows( ...
        visualLoadingTable,'LoadingMagnitude','descend');
    writetable(visualLoadingTable,fullfile(save_dir, ...
        'PLSDA_SortedTrain_2component_loadings.csv'));
else
    warning('Fewer than two PLS components are available for visualization.');
end

%% ============================================================
% VIP analysis from the final positive-sorted training model
%
% VIP feature selection uses:
%   - positive-sorted cells only,
%   - the component count selected by sorted-cell batch CV,
%   - the final model already fitted above.
%
% Pure cells are used only to examine whether the selected features
% retain the same class-associated direction in the external population.
% =============================================================
Wvip = stats.W(:,1:optimalComponents);
Tvip = XS(:,1:optimalComponents);
Qvip = YL;

% MATLAB normally returns YL as responses x components.
if size(Qvip,2) ~= optimalComponents && ...
        size(Qvip,1) == optimalComponents
    Qvip = Qvip';
end

if size(Qvip,2) ~= optimalComponents
    error('Unexpected Y-loading dimensions while calculating VIP scores.');
end

nPredictors = size(Wvip,1);
SSYbyComponent = zeros(1,optimalComponents);
normalizedWeights = zeros(size(Wvip));

for a = 1:optimalComponents
    SSYbyComponent(a) = ...
        sum(Tvip(:,a).^2) * sum(Qvip(:,a).^2);

    weightNorm = norm(Wvip(:,a));
    if weightNorm > 0
        normalizedWeights(:,a) = Wvip(:,a)/weightNorm;
    end
end

VIP = zeros(nPredictors,1);
totalSSY = sum(SSYbyComponent);

if totalSSY <= 0
    warning('The fitted model has zero explained Y sum of squares; VIP is undefined.');
    VIP(:) = NaN;
else
    for j = 1:nPredictors
        VIP(j) = sqrt( ...
            nPredictors * ...
            sum(SSYbyComponent .* normalizedWeights(j,:).^2) / ...
            totalSSY);
    end
end

% Select and rank the requested top fraction.
nTopVIP = max(1,ceil(topVIPFraction*nPredictors));
[VIPsorted,VIPorder] = sort(VIP,'descend');
topVIPRows = VIPorder(1:nTopVIP);
topVIPScores = VIPsorted(1:nTopVIP);
topVIPFeatures = loadingFeatureLabels(topVIPRows);

%% Group means for sorted and pure cells
sortedGroupMeans = NaN(nTopVIP,nClasses);
untouchedGroupMeans = NaN(nTopVIP,nClasses);

for c = 1:nClasses
    sortedClassRows = YtrainIndex == c;
    untouchedClassRows = YtestIndex == c;

    if any(sortedClassRows)
        sortedGroupMeans(:,c) = ...
            mean(Xtrain(sortedClassRows,topVIPRows),1)';
    end

    if any(untouchedClassRows)
        untouchedGroupMeans(:,c) = ...
            mean(Xtest(untouchedClassRows,topVIPRows),1)';
    end
end

% Determine whether the class with the highest mean is consistent between
% the sorted training population and pure external population.
[~,highestSortedClass] = max(sortedGroupMeans,[],2);
[~,highestUntouchedClass] = max(untouchedGroupMeans,[],2);
consistentDirection = highestSortedClass == highestUntouchedClass;

fprintf('\nVIP feature comparison\n');
fprintf('Top fraction: %.1f%%\n',100*topVIPFraction);
fprintf('Number of selected features: %d\n',nTopVIP);
fprintf('Consistent class direction: %d/%d features\n', ...
    sum(consistentDirection),nTopVIP);

%% Export complete VIP table
allVIPTable = table( ...
    loadingFeatureLabels,VIP, ...
    'VariableNames',{'Feature','VIP'});

if useNumericMZAxis
    allVIPTable.MZ = loadingXAxis;
    allVIPTable = movevars(allVIPTable,'MZ','After','Feature');
end

allVIPTable = sortrows(allVIPTable,'VIP','descend');
writetable(allVIPTable,fullfile(save_dir, ...
    'PLSDA_SortedTrain_all_VIP_scores.csv'));

%% Export top-VIP comparison table
topVIPTable = table( ...
    (1:nTopVIP)',topVIPFeatures,topVIPScores,consistentDirection, ...
    'VariableNames',{'Rank','Feature','VIP','ConsistentDirection'});

if useNumericMZAxis
    topVIPTable.MZ = loadingXAxis(topVIPRows);
    topVIPTable = movevars(topVIPTable,'MZ','After','Feature');
end

for c = 1:nClasses
    sortedMeanName = matlab.lang.makeValidName( ...
        "SortedMean_" + classNames(c));
    untouchedMeanName = matlab.lang.makeValidName( ...
        "PureMean_" + classNames(c));

    topVIPTable.(sortedMeanName) = sortedGroupMeans(:,c);
    topVIPTable.(untouchedMeanName) = untouchedGroupMeans(:,c);
end

writetable(topVIPTable,fullfile(save_dir, ...
    'PLSDA_SortedTrain_top_VIP_comparison.csv'));

%% VIP dot plot and sorted-versus-pure heatmap
VIPheatmapRaw = [sortedGroupMeans,untouchedGroupMeans];
VIPheatmapZ = NaN(size(VIPheatmapRaw));

for row = 1:nTopVIP
    rowValues = VIPheatmapRaw(row,:);
    rowStandardDeviation = std(rowValues,0,'omitnan');

    if isfinite(rowStandardDeviation) && rowStandardDeviation > 0
        VIPheatmapZ(row,:) = ...
            (rowValues-mean(rowValues,'omitnan')) / rowStandardDeviation;
    else
        VIPheatmapZ(row,:) = 0;
    end
end

VIPcolumnLabels = [ ...
    classNames(:)' + trainingPlotSuffix, ...
    classNames(:)' + testPlotSuffix];

figVIP = figure('Color','w','Position',[100 100 1100 850]);
vipLayout = tiledlayout(figVIP,1,2, ...
    'TileSpacing','compact','Padding','compact');

% Left: VIP dot plot
axVIP = nexttile(vipLayout,1);
scatter(axVIP,topVIPScores,nTopVIP:-1:1,50, ...
    [0.20 0.50 0.80],'filled');
yticks(axVIP,1:nTopVIP);
yticklabels(axVIP,flip(topVIPFeatures));
xlabel(axVIP,'VIP score');
ylabel(axVIP,'Feature');
title(axVIP,sprintf('Top %.1f%% VIP features',100*topVIPFraction));
ylim(axVIP,[0.5 nTopVIP+0.5]);
box(axVIP,'on');
set(axVIP,'FontSize',plotFontSize,'LineWidth',1.2);

% Right: row-standardized group means
axHeatmap = nexttile(vipLayout,2);
imagesc(axHeatmap,VIPheatmapZ);

nColorSteps = 128;
blueWhiteRed = [ ...
    linspace(0,1,nColorSteps)', ...
    linspace(0,1,nColorSteps)', ...
    ones(nColorSteps,1); ...
    ones(nColorSteps,1), ...
    linspace(1,0,nColorSteps)', ...
    linspace(1,0,nColorSteps)'];

colormap(axHeatmap,blueWhiteRed);
colorbarHandle = colorbar(axHeatmap);
colorbarHandle.Label.String = ...
    'Row-standardized mean intensity (z-score)';
xticks(axHeatmap,1:numel(VIPcolumnLabels));
xticklabels(axHeatmap,VIPcolumnLabels);
xtickangle(axHeatmap,30);
yticks(axHeatmap,1:nTopVIP);
yticklabels(axHeatmap,topVIPFeatures);
title(axHeatmap,'Training and external-test class means');
box(axHeatmap,'on');
set(axHeatmap,'FontSize',plotFontSize,'LineWidth',1.2);

standardizeFigureFonts(figVIP,plotFontSize);

exportgraphics(figVIP,fullfile(save_dir, ...
    'PLSDA_SortedTrain_top_VIP_plot.png'),'Resolution',300);
exportgraphics(figVIP,fullfile(save_dir, ...
    'PLSDA_SortedTrain_top_VIP_plot.pdf'),'Resolution',300);
saveas(figVIP,fullfile(save_dir, ...
    'PLSDA_SortedTrain_top_VIP_plot.fig'));

%% ============================================================
% ID-annotated VIP plot
%
% Only top-VIP features with:
%   1. a numeric m/z value,
%   2. an annotation match within the requested ppm tolerance, and
%   3. a non-empty ID
% are included in this plot and its exported table.
% =============================================================
if strlength(vipAnnotationFile) > 0 && ...
        isfile(vipAnnotationFile) && useNumericMZAxis

    vipAnnotation = readtable(vipAnnotationFile, ...
        'VariableNamingRule','preserve');

    annotationVariableNames = string( ...
        vipAnnotation.Properties.VariableNames);

    if ~any(annotationVariableNames == vipAnnotationMZColumn)
        warning('Annotation m/z column "%s" was not found.', ...
            vipAnnotationMZColumn);

    elseif ~any(annotationVariableNames == vipAnnotationIDColumn)
        warning('Annotation ID column "%s" was not found.', ...
            vipAnnotationIDColumn);

    else
        annotationMZ = vipAnnotation.(vipAnnotationMZColumn);
        annotationIDs = string( ...
            vipAnnotation.(vipAnnotationIDColumn));

        if ~isnumeric(annotationMZ)
            annotationMZ = str2double(string(annotationMZ));
        end

        annotationIDs = strtrim(annotationIDs);

        validAnnotationRows = ...
            isfinite(annotationMZ) & ...
            ~ismissing(annotationIDs) & ...
            annotationIDs ~= "" & ...
            lower(annotationIDs) ~= "nan" & ...
            lower(annotationIDs) ~= "missing";

        annotationMZ = double(annotationMZ(validAnnotationRows));
        annotationIDs = annotationIDs(validAnnotationRows);

        topVIPMZ = loadingXAxis(topVIPRows);
        matchedID = strings(nTopVIP,1);
        matchedAnnotationMZ = NaN(nTopVIP,1);
        matchedErrorPPM = NaN(nTopVIP,1);
        hasAcceptedID = false(nTopVIP,1);

        for row = 1:nTopVIP
            if isempty(annotationMZ) || ...
                    ~isfinite(topVIPMZ(row)) || topVIPMZ(row) == 0
                continue
            end

            ppmErrors = ...
                abs(annotationMZ-topVIPMZ(row)) ./ topVIPMZ(row) * 1e6;

            [minimumPPM,closestAnnotationRow] = min(ppmErrors);

            if minimumPPM <= vipAnnotationTolerancePPM
                matchedID(row) = annotationIDs(closestAnnotationRow);
                matchedAnnotationMZ(row) = ...
                    annotationMZ(closestAnnotationRow);
                matchedErrorPPM(row) = minimumPPM;
                hasAcceptedID(row) = true;
            end
        end

        fprintf('\nVIP annotation matching\n');
        fprintf('Annotation tolerance: %.2f ppm\n', ...
            vipAnnotationTolerancePPM);
        fprintf('Annotated top VIP features: %d/%d\n', ...
            sum(hasAcceptedID),nTopVIP);

        if any(hasAcceptedID)
            annotatedRows = find(hasAcceptedID);

            annotatedIDs = matchedID(annotatedRows);
            annotatedFeatureMZ = topVIPMZ(annotatedRows);
            annotatedReferenceMZ = matchedAnnotationMZ(annotatedRows);
            annotatedErrorPPM = matchedErrorPPM(annotatedRows);
            annotatedVIP = topVIPScores(annotatedRows);
            annotatedConsistency = consistentDirection(annotatedRows);
            annotatedHeatmapZ = VIPheatmapZ(annotatedRows,:);

            % Include m/z in labels so repeated IDs remain distinguishable.
            annotatedPlotLabels = annotatedIDs + ...
                " (" + compose('%.4f',annotatedFeatureMZ) + ")";

            nAnnotatedVIP = numel(annotatedRows);

            annotatedFigureHeight = max(850,70*nAnnotatedVIP);
            figAnnotatedVIP = figure( ...
                'Color','w','Position',[50 50 1500 annotatedFigureHeight]);

            annotatedLayout = tiledlayout( ...
                figAnnotatedVIP,1,2, ...
                'TileSpacing','compact','Padding','compact');

            % Left: VIP dot plot containing annotated features only
            axAnnotatedVIP = nexttile(annotatedLayout,1);
            scatter(axAnnotatedVIP,annotatedVIP, ...
                nAnnotatedVIP:-1:1,55, ...
                [0.20 0.50 0.80],'filled');
            yticks(axAnnotatedVIP,1:nAnnotatedVIP);
            yticklabels(axAnnotatedVIP,flip(annotatedPlotLabels));
            xlabel(axAnnotatedVIP,'VIP score');
            ylabel(axAnnotatedVIP,'Annotated lipid feature');
            title(axAnnotatedVIP,sprintf( ...
                'Annotated features among top %.1f%% VIPs', ...
                100*topVIPFraction));
            ylim(axAnnotatedVIP,[0.5 nAnnotatedVIP+0.5]);
            box(axAnnotatedVIP,'on');
            set(axAnnotatedVIP,'FontSize',plotFontSize,'LineWidth',1.2);

            % Right: same sorted-versus-pure heatmap
            axAnnotatedHeatmap = nexttile(annotatedLayout,2);
            imagesc(axAnnotatedHeatmap,annotatedHeatmapZ);
            colormap(axAnnotatedHeatmap,blueWhiteRed);
            annotatedColorbar = colorbar(axAnnotatedHeatmap);
            annotatedColorbar.Label.String = ...
                'Row-standardized mean intensity (z-score)';
            xticks(axAnnotatedHeatmap,1:numel(VIPcolumnLabels));
            xticklabels(axAnnotatedHeatmap,VIPcolumnLabels);
            xtickangle(axAnnotatedHeatmap,30);
            yticks(axAnnotatedHeatmap,1:nAnnotatedVIP);
            yticklabels(axAnnotatedHeatmap,[]);
            title(axAnnotatedHeatmap, ...
                'Training and external-test class means');
            box(axAnnotatedHeatmap,'on');
            set(axAnnotatedHeatmap,'FontSize',plotFontSize,'LineWidth',1.2);

            standardizeFigureFonts(figAnnotatedVIP,plotFontSize);

            exportgraphics(figAnnotatedVIP,fullfile(save_dir, ...
                'PLSDA_SortedTrain_top_VIP_annotated_IDs.png'), ...
                'Resolution',300);
            exportgraphics(figAnnotatedVIP,fullfile(save_dir, ...
                'PLSDA_SortedTrain_top_VIP_annotated_IDs.pdf'), ...
                'Resolution',300);
            saveas(figAnnotatedVIP,fullfile(save_dir, ...
                'PLSDA_SortedTrain_top_VIP_annotated_IDs.fig'));

            % Export annotated features and their group means
            annotatedVIPTable = table( ...
                (1:nAnnotatedVIP)',annotatedIDs, ...
                annotatedFeatureMZ,annotatedReferenceMZ, ...
                annotatedErrorPPM,annotatedVIP,annotatedConsistency, ...
                'VariableNames',{ ...
                'Rank','ID','FeatureMZ','AnnotationMZ','MassErrorPPM', ...
                'VIP','ConsistentDirection'});

            for c = 1:nClasses
                sortedMeanName = matlab.lang.makeValidName( ...
                    "SortedMean_" + classNames(c));
                untouchedMeanName = matlab.lang.makeValidName( ...
                    "PureMean_" + classNames(c));

                annotatedVIPTable.(sortedMeanName) = ...
                    sortedGroupMeans(annotatedRows,c);
                annotatedVIPTable.(untouchedMeanName) = ...
                    untouchedGroupMeans(annotatedRows,c);
            end

            writetable(annotatedVIPTable,fullfile(save_dir, ...
                'PLSDA_SortedTrain_top_VIP_annotated_IDs.csv'));
        else
            warning(['None of the top VIP features had a non-empty ID ' ...
                'within %.2f ppm.'],vipAnnotationTolerancePPM);
        end
    end

elseif strlength(vipAnnotationFile) == 0
    warning('vipAnnotationFile is empty; the ID-annotated VIP plot was skipped.');
elseif ~isfile(vipAnnotationFile)
    warning('VIP annotation file was not found: %s',vipAnnotationFile);
elseif ~useNumericMZAxis
    warning(['Numeric MZlist values are required for the ID-annotated ' ...
        'VIP plot.']);
end

%% Exact batch-blocked permutation test
%
% For two classes, independently swap the B/T labels within each complete
% positive-sorted batch. With three batches, only 2^3 = 8 distinct label
% configurations exist, including the observed configuration.
%
% Every permutation repeats:
%   component selection using sorted batches only,
%   final fitting using all sorted cells,
%   prediction of the pure test population.

if nClasses == 2
    nConfigurations = 2^nSortedBatches;
    permutationAccuracy = NaN(nConfigurations,1);
    permutationBalancedAccuracy = NaN(nConfigurations,1);
    permutationQ2 = NaN(nConfigurations,1);
    permutationComponents = NaN(nConfigurations,1);

    for configuration = 0:(nConfigurations-1)
        swapPattern = logical(bitget(configuration,1:nSortedBatches));
        permutedTrainIndex = YtrainIndex;

        for b = 1:nSortedBatches
            if swapPattern(b)
                rowsInBatch = trainBatches == sortedBatchNames(b);
                permutedTrainIndex(rowsInBatch) = ...
                    3 - permutedTrainIndex(rowsInBatch);
            end
        end

        permutedYtrain = zeros(size(Ytrain));
        permutedYtrain(:,1) = permutedTrainIndex == 1;
        permutedYtrain(:,2) = permutedTrainIndex == 2;

        [permutedComponents,~,~,~,~] = selectComponentsByBatch( ...
            XtrainRaw, permutedYtrain, permutedTrainIndex, ...
            trainBatches, maximumComponents);

        permutationComponents(configuration+1) = permutedComponents;

        [~,~,~,~,BETA_permuted] = plsregress( ...
            Xtrain, permutedYtrain, permutedComponents);

        YtestPermutation = ...
            [ones(size(Xtest,1),1), Xtest] * BETA_permuted;

        [~, predictedPermutation] = max(YtestPermutation,[],2);

        permutationAccuracy(configuration+1) = ...
            mean(predictedPermutation == YtestIndex);

        cmPermutation = confusionmat( ...
            YtestIndex, predictedPermutation, 'Order',1:nClasses);
        recallPermutation = diag(cmPermutation) ./ ...
            sum(cmPermutation,2);
        permutationBalancedAccuracy(configuration+1) = ...
            mean(recallPermutation,'omitnan');

        PRESS_permutation = sum( ...
            (Ytest - YtestPermutation).^2,'all');
        permutationQ2(configuration+1) = ...
            1 - PRESS_permutation/TSS_external;
    end

    % Exact randomization p-values. The observed configuration is included.
    permutationPAccuracy = mean( ...
        permutationBalancedAccuracy >= externalBalancedAccuracy);
    permutationPQ2 = mean(permutationQ2 >= externalQ2);

    fprintf('\nExact batch-blocked permutation test\n');
    fprintf('Distinct configurations: %d\n', nConfigurations);
    fprintf('Balanced-accuracy p-value: %.4f\n', permutationPAccuracy);
    fprintf('Q2 p-value: %.4f\n', permutationPQ2);
    fprintf(['With %d batches, the smallest attainable exact p-value ' ...
        'is %.4f.\n'], nSortedBatches,1/nConfigurations);

    permutationTable = table( ...
        (0:nConfigurations-1)', permutationComponents, ...
        permutationAccuracy, permutationBalancedAccuracy, permutationQ2, ...
        'VariableNames',{ ...
        'Configuration','SelectedComponents','Accuracy', ...
        'BalancedAccuracy','Q2'});

    writetable(permutationTable, fullfile(save_dir, ...
        'PLSDA_Training_to_ExternalTest_batch_permutations.csv'));

    figPermutation = figure('Color','w','Position',[100 100 750 550]);
    histogram(permutationBalancedAccuracy, ...
        'BinMethod','integers', ...
        'FaceColor',[0.65 0.65 0.65]);
    hold on;
    xline(externalBalancedAccuracy,'r-','LineWidth',2.5);
    hold off;
    xlabel('External balanced accuracy under batch label swaps');
    ylabel('Number of configurations');
    title(sprintf( ...
        'Exact batch-blocked permutation test: p = %.3f', ...
        permutationPAccuracy));
    grid on;

    standardizeFigureFonts(figPermutation,plotFontSize);

    exportgraphics(figPermutation, fullfile(save_dir, ...
        'PLSDA_Training_to_ExternalTest_permutation.png'), ...
        'Resolution',300);
    saveas(figPermutation, fullfile(save_dir, ...
        'PLSDA_Training_to_ExternalTest_permutation.fig'));
else
    warning(['The included batch-swap permutation test is defined for ' ...
        'two classes and was not run.']);
end

fprintf('\nAnalysis completed. Results saved in:\n%s\n',save_dir);

%% Local utility: enforce consistent figure typography
function standardizeFigureFonts(figHandle,fontSize)
fontObjects = findall(figHandle,'-property','FontSize');
for objectIndex = 1:numel(fontObjects)
    try
        fontObjects(objectIndex).FontSize = fontSize;
    catch
        % Some chart internals expose FontSize as read-only.
    end
end

% Make hierarchy readable after figures are reduced for publication.
axesObjects = findall(figHandle,'Type','axes');
for objectIndex = 1:numel(axesObjects)
    ax = axesObjects(objectIndex);
    try
        ax.FontSize = fontSize;           % tick numbers
        ax.LineWidth = 1.4;
        ax.XLabel.FontSize = fontSize+2;
        ax.YLabel.FontSize = fontSize+2;
        ax.ZLabel.FontSize = fontSize+2;
        ax.Title.FontSize = fontSize+3;
        ax.Title.FontWeight = 'bold';
    catch
    end
end

legendObjects = findall(figHandle,'Type','legend');
for objectIndex = 1:numel(legendObjects)
    try
        legendObjects(objectIndex).FontSize = max(12,fontSize-1);
    catch
    end
end

fontNameObjects = findall(figHandle,'-property','FontName');
for objectIndex = 1:numel(fontNameObjects)
    try
        fontNameObjects(objectIndex).FontName = 'Arial';
    catch
    end
end
end

%% ============================================================
% Local function: component selection using sorted batches only
% =============================================================
function [optimalComponents, accuracy, balancedAccuracy, Q2, predictions] = ...
    selectComponentsByBatch(Xraw,Y,Yindex,batches,maximumComponents)

batchNames = unique(batches,'stable');
nBatches = numel(batchNames);
nClasses = size(Y,2);

maxComponents = min([ ...
    maximumComponents, ...
    size(Xraw,1)-1, ...
    size(Xraw,2), ...
    rank(Xraw)]);
maxComponents = max(1,maxComponents);

accuracy = NaN(1,maxComponents);
balancedAccuracy = NaN(1,maxComponents);
Q2 = NaN(1,maxComponents);
predictions = cell(1,maxComponents);

for nComp = 1:maxComponents
    Ypred = NaN(size(Y));

    for fold = 1:nBatches
        validationRows = batches == batchNames(fold);
        trainingRows = ~validationRows;

        XfoldTrainRaw = Xraw(trainingRows,:);
        XfoldValidationRaw = Xraw(validationRows,:);
        YfoldTrain = Y(trainingRows,:);

        if numel(unique(Yindex(trainingRows))) < nClasses
            continue
        end

        scale90 = prctile(XfoldTrainRaw,90,1);
        scale90(~isfinite(scale90) | scale90 == 0) = 1;

        XfoldTrain = XfoldTrainRaw ./ scale90;
        XfoldValidation = XfoldValidationRaw ./ scale90;

        maximumFoldComponents = min([ ...
            size(XfoldTrain,1)-1, ...
            size(XfoldTrain,2), ...
            rank(XfoldTrain)]);

        if nComp > maximumFoldComponents
            continue
        end

        [~,~,~,~,BETAfold] = plsregress( ...
            XfoldTrain,YfoldTrain,nComp);

        Ypred(validationRows,:) = ...
            [ones(sum(validationRows),1),XfoldValidation] * BETAfold;
    end

    if ~all(isfinite(Ypred),'all')
        continue
    end

    predictions{nComp} = Ypred;
    [~,predictedClass] = max(Ypred,[],2);

    accuracy(nComp) = mean(predictedClass == Yindex);

    cm = confusionmat(Yindex,predictedClass,'Order',1:nClasses);
    recall = diag(cm) ./ sum(cm,2);
    balancedAccuracy(nComp) = mean(recall,'omitnan');

    PRESS = sum((Y-Ypred).^2,'all');
    TSS = sum((Y-mean(Y,1)).^2,'all');

    if TSS > 0
        Q2(nComp) = 1 - PRESS/TSS;
    end
end

validComponents = find(isfinite(balancedAccuracy));

if isempty(validComponents)
    error('No component count produced complete batch-held-out predictions.');
end

bestBalancedAccuracy = max(balancedAccuracy(validComponents));
tolerance = 1e-12;

candidates = validComponents( ...
    abs(balancedAccuracy(validComponents)-bestBalancedAccuracy) < tolerance);

if numel(candidates) > 1
    candidateQ2 = Q2(candidates);

    if any(isfinite(candidateQ2))
        bestQ2 = max(candidateQ2,[],'omitnan');
        candidates = candidates( ...
            abs(candidateQ2-bestQ2) < tolerance);
    end
end

% Prefer the simplest equally performing model.
optimalComponents = min(candidates);
end
