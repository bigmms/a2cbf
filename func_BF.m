function Iout = func_BF(Isrc, Rad, StdS, StdR)

[Hei,Wid] = size(Isrc);
iKg = zeros(1,2*Rad+1);
for i = 0 : 1 : Rad
    iKg(Rad+i+1) = exp(-(i^2)/2/(StdS^2));
    iKg(Rad-i+1) = exp(-(i^2)/2/(StdS^2));
end
iKs = iKg' * iKg;
iKr = zeros(1,256);
for i = 0 : 1 : 255
    iKr(i+1) = exp(-(i^2)/2/(StdR^2));
end

Iout = Isrc;
for h = 1 : 1 : Hei
    for w = 1 : 1 : Wid
        SumUp = 0;
        SumDn = 0;
        for sh = -Rad : 1 : Rad
            i = min(max(h + sh, 1), Hei);
            for sw = -Rad : 1 : Rad
                j = min(max(w + sw, 1), Wid);
                PxlDif = abs(Isrc(i,j) - Isrc(h,w));
                SumUp = SumUp + Isrc(i,j) * iKs(sh+Rad+1, sw+Rad+1) * iKr(PxlDif+1);
                SumDn = SumDn + iKs(sh+Rad+1, sw+Rad+1) * iKr(PxlDif+1);
            end
        end
        Iout(h,w) = max(min(round(SumUp / SumDn), 255), 0);
    end
end

end