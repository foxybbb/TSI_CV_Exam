% Test file to verify SURF feature extraction

% Read the input image
imageFile = '1.jpg'; % Replace with your image file name
inputImage = imread(imageFile);

% Convert the image to grayscale if it's RGB
if size(inputImage, 3) == 3
    inputImage = rgb2gray(inputImage);
end

% Detect the face using a pre-trained Viola-Jones face detector
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
faceBbox = step(faceDetector, inputImage);

% Initialize the feature vector
featureVector = [];

% Ensure a face is detected
if ~isempty(faceBbox)
    % Crop the first detected face
    faceROI = imcrop(inputImage, faceBbox(1, :));

    % Display the cropped face
    figure;
    imshow(faceROI);
    title('Cropped Face');

    % Extract SURF keypoints and descriptors
    points = detectSURFFeatures(faceROI);
    [features, validPoints] = extractFeatures(faceROI, points);

    % Visualize the strongest SURF keypoints
    figure;
    imshow(faceROI);
    hold on;
    plot(validPoints.selectStrongest(20));
    title('Strongest SURF Keypoints');
    hold off;

    % Use the mean of the feature vectors as a compact representation
    if ~isempty(features)
        meanKeypointFeatures = mean(features.Features, 1);
    else
        meanKeypointFeatures = zeros(1, 64); % Ensure compatibility if no keypoints
    end

    % Add keypoint features to the feature vector
    featureVector = [featureVector, meanKeypointFeatures];

    % Display the length of the feature vector
    fprintf('Extracted SURF features. Feature vector length: %d\n', length(featureVector));
else
    fprintf('No face detected, SURF feature extraction skipped.\n');
end
