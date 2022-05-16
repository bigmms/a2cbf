function Iout = func_imnoise(hei, wid, sig, type)

if type == 0 % Gaussian noise
    Iout = randn(hei, wid) .* sig;
elseif type == 1 % uniform noise
    Iout = 2 * (randn(hei, wid) - 0.5) * (sqrt(3) * sig);
else % impulse noise
    Nimp = log(rand(hei, wid)).*sign(randn(hei, wid));
    Iout = Nimp / std(Nimp(:)) * sig;
end

end