% ========== AZIMUTHALAVERAGE.M ==========
function azProfile = azimuthalAverage(powerSpectrum)
% azimuthalAverage:
%   Compute the radial (azimuthal) average of a 2D power spectrum.
%
    [rows, cols] = size(powerSpectrum);
    cx = floor(cols/2) + 1;
    cy = floor(rows/2) + 1;

    [X, Y] = meshgrid(1:cols, 1:rows);
    R = sqrt((X - cx).^2 + (Y - cy).^2);
    R = round(R);

    maxR = floor(min(rows, cols) / 2);
    azProfile = zeros(1, maxR + 1);
    counts    = zeros(1, maxR + 1);

    for r = 0:maxR
        mask = (R == r);
        azProfile(r+1) = sum(powerSpectrum(mask));
        counts(r+1)    = sum(mask(:));
    end

    azProfile = azProfile ./ (counts + eps);
end
