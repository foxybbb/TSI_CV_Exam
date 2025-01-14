function [svmStruct, featureMatrix, labelVector] = train_spoof_detector(dataFolder)
% train_spoof_detector:
%   1) Iterates over labeled images (real vs. fake) in 'dataFolder'
%   2) Detects the main face, crops/resizes it
%   3) Extracts features 
%   4) Train
%   5) Returns the SVM struct
%   6) Includes feature visualization plots

    % --- 1. Setup subfolders for real vs. fake
    realFolder = fullfile(dataFolder, 'real');
    fakeFolder = fullfile(dataFolder, 'fake');

    realImages = dir(fullfile(realFolder, '*.jpg'));
    fakeImages = dir(fullfile(fakeFolder, '*.jpg'));

    % Holders for features & labels
    featureMatrix = [];
    labelVector   = [];

    % --- 2. Process REAL images
    fprintf('Processing REAL images...\n');
    for i = 1:length(realImages)
        imagePath = fullfile(realFolder, realImages(i).name);
        [faceFeatures, success] = extract_face_features(imagePath);
        if success
            featureMatrix = [featureMatrix; faceFeatures];
            labelVector   = [labelVector; 1];  % "1" = Real
            fprintf('  Processed REAL image %d/%d: %s\n', i, length(realImages), realImages(i).name);
        else
            fprintf('  Skipped REAL image %d/%d (no face found): %s\n', i, length(realImages), realImages(i).name);
        end
    end

    % --- 3. Process FAKE images
    fprintf('Processing FAKE images...\n');
    for i = 1:length(fakeImages)
        imagePath = fullfile(fakeFolder, fakeImages(i).name);
        [faceFeatures, success] = extract_face_features(imagePath);
        if success
            featureMatrix = [featureMatrix; faceFeatures];
            labelVector   = [labelVector; 0];  % "0" = Fake
            fprintf('  Processed FAKE image %d/%d: %s\n', i, length(fakeImages), fakeImages(i).name);
        else
            fprintf('  Skipped FAKE image %d/%d (no face found): %s\n', i, length(fakeImages), fakeImages(i).name);
        end
    end

    % --- 4. Train SVM using older MATLAB syntax
    fprintf('Training SVM model...\n');
    svmStruct = svmtrain(featureMatrix, labelVector, ...
                         'Kernel_Function', 'linear', ...
                         'Autoscale', true);

    % --- 5. Visualizations
    visualize_features(featureMatrix, labelVector, svmStruct);
end

% --- Visualization Function ---

% --- Visualization Function ---
function visualize_features(featureMatrix, labelVector, svmStruct)
    % 2. Plot (Scatter for first two features)
    figure;
    scatter(featureMatrix(labelVector == 1, 1), featureMatrix(labelVector == 1, 2), 'b', 'filled');
    hold on;
    scatter(featureMatrix(labelVector == 0, 1), featureMatrix(labelVector == 0, 2), 'r', 'filled');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('Real', 'Fake');
    title('Feature Space (First 2 Features)');
    hold off;

    % 3. Correlation Heatmap 
    figure;
    corrMatrix = corr(featureMatrix);
    imagesc(corrMatrix);
    colorbar;
    title('Feature Correlation Heatmap');
    xlabel('Features');
    ylabel('Features');

    % Linear SVM Coeficens: 
    sv = svmStruct.SupportVectors;
    alpha = svmStruct.Alpha;
    % Weight for classficatior
    w = sv' * alpha;
    % Abs Values
    absW = abs(w);

    % Sort by decrease value 
    [sortedAbsW, sortIdx] = sort(absW, 'descend');

    % Printing
    sortedW = w(sortIdx);
     for i = 1:length(sortedW)
    fprintf('Feature %d: weight = %.4f\n', sortIdx(i), sortedW(i));
    end


end

