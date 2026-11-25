# Kozmoloji

Hedef modül: `kozmoloji/`. Uçtan uca uygulaması: [../08_ornek_projects/01_hubble_diyagrami.md](../08_ornek_projects/01_hubble_diyagrami.md).

Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. Genişleyen evren

- Friedmann-Lemaître-Robertson-Walker metriği, ölçek çarpanı `a(t)`, kırmızıya kayma `1 + z = 1/a`.
- Friedmann denklemi ve boyutsuz Hubble parametresi [Hogg1999, Denk. 14; CarrollPressTurner1992]:

  `E(z) = H(z)/H0 = sqrt( Ωr(1+z)⁴ + Ωm(1+z)³ + Ωk(1+z)² + Ω_DE f(z) )`

  `Ωk = 1 − Ωm − Ωr − Ω_DE`. Kozmolojik sabit için `f = 1`; sabit `w` için `f = (1+z)^{3(1+w)}`; `w(a) = w0 + wa(1 − a)` parametrelemesi için `f = (1+z)^{3(1+w0+wa)} exp(−3 wa z/(1+z))` [ChevallierPolarski2001; Linder2003].
- Radyasyon yoğunluğu: fotonlar CMB sıcaklığından, nötrinolar etkin tür sayısından; kütleli nötrinolar ayrı terimdir ve ilk aşamada kütlesiz alınır.
- Hubble uzaklığı `D_H = c/H0` ve Hubble zamanı `t_H = 1/H0`.
- Kırmızıya kayma, uzaklaşma hızı ve ufuk kavramlarının yaygın yanlış yorumları [DavisLineweaver2004].

## 2. Uzaklık ölçüleri

Tümü Hogg'un derlemesindeki tanımları izler [Hogg1999]:

- Eşyürüyen uzaklık (bakış doğrultusu): `D_C = D_H ∫₀ᶻ dz′/E(z′)` (Denk. 15)
- Enine eşyürüyen uzaklık `D_M`: `Ωk > 0` için `(D_H/√Ωk) sinh(√Ωk D_C/D_H)`, `Ωk = 0` için `D_C`, `Ωk < 0` için `sin` biçimi (Denk. 16)
- Açısal çap uzaklığı: `D_A = D_M/(1+z)` (Denk. 18)
- Işınım gücü uzaklığı: `D_L = (1+z) D_M` (Denk. 21)
- Uzaklık modülü: `μ = 5 log10(D_L / 10 pc)` (Denk. 25)
- Eşyürüyen hacim elemanı (Denk. 28), geriye bakış zamanı (Denk. 30), evrenin yaşı
- İki kırmızıya kayma arasındaki açısal çap uzaklığı (merceklenme için, Denk. 19)

Denklem numaraları Hogg'un arXiv sürümünden hatırlanmıştır ve kodlama sırasında teyit edilecektir.

Düşük `z` açılımı: `D_L ≈ (c z/H0) [1 + (1 − q0) z/2 + …]`; yavaşlama parametresi `q0` ve sarsım `j0` ile kozmografik açılım [Riess2022, Denk. 5 dolayı]. Sayısal integralin `z → 0` limitinde bu açılıma yakınsaması özellik testidir.

Kâhin: `astropy.cosmology`. Bağımsız karşılaştırma: Wright'ın hesaplayıcısının yöntemi [Wright2006].

## 3. Hubble sabitinin ölçüm yolları

**Yerel uzaklık merdiveni.** Üç basamak [Riess2022]:

1. Geometrik çapa: Samanyolu Sefeidlerinin Gaia paralaksları, Büyük Macellan Bulutu'nun örten çift uzaklığı, NGC 4258'in mazer uzaklığı.
2. Birincil gösterge: Sefeid dönem-ışınım gücü bağıntısıyla Tür Ia süpernova konak gökadalarının uzaklığı; almaşık olarak TRGB ve JAGB [Freedman2019; Freedman2025].
3. Hubble akışı: standartlaştırılmış Tür Ia süpernovaların `0,023 < z < 0,15` aralığındaki kadir-kırmızıya kayma bağıntısı.

`H0`, süpernovaların mutlak kadiri `M_B` ile tam dejeneredir: Hubble diyagramı tek başına yalnızca `H0` ve `M_B` birleşimini verir, mutlak ölçek birinci ve ikinci basamaktan gelir. Bu dejenerasyon projede açıkça gösterilir.

**Ters uzaklık merdiveni ve erken evren.** CMB güç tayfından ΛCDM varsayımıyla türetilen `H0` [Planck2020]; BAO ölçümlerinin ses ufkuyla kalibrasyonu [DESI2025; DESI2025DR2].

**Bağımsız yollar.** Çekimsel dalga standart sirenleri [Abbott2017], mercek zaman gecikmeleri, kozmik kronometreler (`H(z)` doğrudan) [Moresco2022].

**Güncel durum.** Araştırma sırasında okunan değerler:

- SH0ES Sefeid merdiveni: `H0 = 73,04 ± 1,04 km s⁻¹ Mpc⁻¹` [Riess2022]. Değer makalenin arXiv özet sayfasından okundu.
- CCHP, yalnızca TRGB: `70,39 ± 1,22 (ist.) ± 1,33 (sis.) ± 0,70 (σ_SN)` [Freedman2025]. Değer arXiv:2408.06153 özet sayfasından okundu; makalenin sürümleri arasında değer değişmiştir, kullanılacak sürüm kaydedilmelidir.
- Planck 2018 ΛCDM değeri (yaklaşık 67,4 ± 0,5) bu oturumda kaynağından okunmamıştır [Planck2020].

Yerel ve erken evren değerleri arasındaki fark "Hubble gerilimi" olarak anılır; çözüm önerilerinin derlemesi [DiValentino2021]. Sefeid ve TRGB tabanlı merdivenler arasındaki fark da açık bir tartışmadır. 2025-2026 yayınlarının (CosmoVerse raporu, DESI DR2, Union3.1, DES-SN5YR yeniden kalibrasyonu) taranması yapılacaklar listesindedir; araştırmada bunların yalnızca arama özetleri görülmüştür.

## 4. Tür Ia süpernovaların standartlaştırılması

- Parlaklık-sönme hızı bağıntısı [Phillips1993] ve renk düzeltmesi [Tripp1998].
- Tripp bağıntısı: `μ = m_B − M_B + α x1 − β c − δ_bias + δ_konak`. `x1` gerilme, `c` renk parametresi; SALT2 ve SALT3 ışık eğrisi modellerinden uydurulur [Guy2007; Kenworthy2021].
- Konak gökada kütle basamağı ve yanlılık düzeltmesi [KesslerScolnic2017; Brout2022].
- Özel hız düzeltmesi: düşük `z`'de kırmızıya kaymanın CMB çerçevesine ve akış modeline göre düzeltilmesi [Carr2022].
- Işık eğrisi uydurma yeniden yazılmaz; `sncosmo` kullanılır ve Pantheon+ derlemesinin yayımlanmış uzaklık modülleri girdi alınır [Scolnic2022].

## 5. Olabilirlik

Kovaryanslı Gauss olabilirliği: `χ² = Δμᵀ C⁻¹ Δμ`, `Δμ_i = μ_i − μ_model(z_i; θ)`; `C` istatistik ve sistematik kovaryansın toplamıdır [Brout2022].

- `M_B` (eşdeğer olarak `H0`) üzerinde analitik marjinalleştirme: düz önselle kapalı biçim [Conley2011, Ek C]. Ek numarası kodlama sırasında teyit edilecek.
- Sefeid çapalı süpernovaların eklenmesi dejenerasyonu kırar: Pantheon+SH0ES veri dosyasında bu süpernovalar işaretlidir ve model değeri yerine Sefeid uzaklığıyla karşılaştırılır [Brout2022].
- Birleşik olabilirlik: SN + BAO + CMB sıkıştırılmış uzaklık önselleri (ileri aşama).

## 6. Erken evren ve yapı oluşumu (ileri aşama)

- Isıl tarih, yeniden birleşme, ses ufku `r_s` (uydurma formülü [EisensteinHu1998]).
- Doğrusal büyüme çarpanı `D(z)` ve büyüme hızı `f ≈ Ωm(z)^0,55`.
- Madde güç tayfı ve aktarım fonksiyonu; Boltzmann kodları (CAMB, CLASS) yeniden yazılmaz, kâhin ve girdi kaynağıdır [Dodelson2020].

## Test durumları

- Einstein-de Sitter evreninde kapalı çözümler: `D_C = 2 D_H [1 − (1+z)^(−1/2)]`, yaş `(2/3) t_H`.
- `Ωk ≠ 0` için `sinh/sin` dallarının `Ωk → 0` limitinde sürekliliği.
- `astropy.cosmology` ile `0 < z < 1100` aralığında bağıl fark `1e-8` altında.
- Hogg'un şekillerinin yeniden üretimi [Hogg1999, Şek. 1-6].
- Pantheon+ ile düz ΛCDM uyumu yayımlanan `Ωm` aralığını vermelidir [Brout2022]; yayımlanan değer kodlama sırasında makaleden okunacak.
