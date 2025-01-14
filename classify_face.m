function [isReal, labelText] = classify_face(svmModel, imagePath)
% classify_face:
    % Extract features from the image
    [faceFeatures, success] = extract_face_features(imagePath);

    if ~success
        error('Could not extract features for face in %s.', imagePath);
    end

    % Perform classificationa
    predictedLabel = svmclassify(svmModel, faceFeatures);

    % Set the result
    if predictedLabel == 1
        isReal = true;
        labelText = 'REAL';
    else
        isReal = false;
        labelText = 'FAKE';
    end
end
