% ========== PREDICT_SPOOF.M ==========
function predict_spoof(svmStruct, testImagePath)
% predict_spoof: 
%   1) Extract face + features from testImagePath
%   2) Classify using 'svmclassify'
%
    [faceFeatures, success] = extract_face_features(testImagePath);
    if ~success
        fprintf('No face detected in %s\n', testImagePath);
        return;
    end

    predictedLabel = svmclassify(svmStruct, faceFeatures);
    if predictedLabel == 1
        disp(['Prediction: REAL face (', testImagePath, ').']);
    else
        disp(['Prediction: FAKE face (', testImagePath, ').']);
    end
end
