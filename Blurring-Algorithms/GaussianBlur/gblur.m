%{
This is a script to model the global blur properties of the 3D dataset.
Note that the blur propogation is abstracted away.

author: ac25 (Arjun Chandra)
%}


%load image
test = imread('cat.jpg');

%convert image to greyscale
testgs = rgb2gray(test);
%image(imgaussfilt(testgs,2))

channel = 1;

%create channels: 1-11 most blur to less (n=sig)
for n = 11:-1:1
    I = imgaussfilt(testgs,2*n);
    blur{channel} = I;
    channel = channel + 1;
end

%channel 12 no blur 
blur{channel} = testgs;
channel = channel + 1;

%13-23 less blur to most (j = sig)
for j = 1:11
    I = imgaussfilt(testgs,2*j);
    blur{channel} = I;
    channel = channel + 1;
end


montage(blur(1:23))

