clear, clc;
imgRoot = './dataset/DIV2K/';
imnames=dir([imgRoot '*' 'png']);
type = 0;
sig = 0.1;
for img = 1 : length(imnames)
    fprintf('%04d\n', img)
    strin = sprintf('%s%04d.png', imgRoot, img);
    Isrc = im2double(imread(strin));
    
    [hei,wid,~] = size(Isrc);
    Iout = func_imnoise(hei, wid, sig, type);
    Iout = max(min(Iout + Isrc, 1), 0);
    
    strin = sprintf('./dataset/testsets/%04d.png', img);
	%strin = sprintf('./dataset/trainsets/%04d.png', img);
	%strin = sprintf('./dataset/valset/%04d.png', img);
    imwrite(Iout, strin);
end
fprintf('done')