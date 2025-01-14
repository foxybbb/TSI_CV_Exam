% Load the test image
testImagePath = 'task.png'; % Replace with your image path
I = imread(testImagePath);

% Convert to grayscale for face detection
if size(I, 3) == 3
    grayImg = rgb2gray(I);
else
    grayImg = I;
end

% Detect faces
faceDetector = vision.CascadeObjectDetector();
bboxes = step(faceDetector, grayImg);

if isempty(bboxes)
    disp('No faces detected in the image.');
    return;
end

% Save each detected face as a PNG file
outputFolder = 'detected_faces'; % Folder to save faces
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder); % Create the folder if it doesn't exist
end

for i = 1:size(bboxes, 1)
    % Crop the face region
    bbox = bboxes(i, :);
    faceROI = imcrop(I, bbox);
   

    % Save the face as a PNG file
    outputFileName = fullfile(outputFolder, sprintf('face_%d.jpg', i));
    imwrite(faceROI, outputFileName);

    fprintf('Saved face %d to %s\n', i, outputFileName);
end
