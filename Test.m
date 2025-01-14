% ========== DETECT_AND_PREDICT.M ==========
load('svmModel')
% Step 2: Load the test image
testImagePath = 'grak5.jpg'; % Replace with your test image path
I = imread(testImagePath);

% Step 3: Convert to grayscale for face detection
if size(I, 3) == 3
    grayImg = rgb2gray(I);
else
    grayImg = I;
end

% Step 4: Detect faces
faceDetector = vision.CascadeObjectDetector();
bboxes = step(faceDetector, grayImg);

if isempty(bboxes)
    disp('No faces detected in the image.');
    return;
end

% Step 5: Display the detected faces and annotate classification
figure; imshow(I); hold on;
title('Detected Faces with Classification');

for i = 1:size(bboxes, 1)
    bbox = bboxes(i, :);

    % Crop the face region
    faceROI = imcrop(I, bbox);

    % Save cropped face to a temporary file
    tempFilePath = sprintf('temp_face_%d.jpg', i);
    imwrite(faceROI, tempFilePath);

    % Call predict_spoof to classify the cropped face
    [isReal, labelText] = classify_face(svmModel, tempFilePath);

    % Draw the bounding box
    rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 2);

    % Add face number and classification label near the bounding box
    text(bbox(1), bbox(2) - 20, sprintf('Face %d: %s', i, labelText), ...
        'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold');

    % Delete the temporary file
    delete(tempFilePath);
end

hold off;
