# İşlev envanteri

Modül modül yazılacak fonksiyonların listesi. Uygulanmış olanların güncel durumu elle tutulmaz; [DURUM.md](DURUM.md) dosyası kaynak kütüğünden üretilir. Bu belgedeki "uygulandı" işaretleri 25 Kasım 2026 tarihli ilk sürüme aittir.

Her fonksiyon satırı adı, girdi ve çıktı birimlerini ve birincil kaynağını verir. Adlar değişebilir; birim ve kaynak sözleşmenin parçasıdır.

## `temel`

**`sabitler`** (uygulandı). CODATA 2022 fiziksel sabitleri [CODATA2022], IAU 2015 B3 nominal Güneş ve gezegen değerleri [Prsa2016], IAU 2015 B2 bolometrik sıfır noktası [Mamajek2015], zaman sabitleri [ERFA2023]. Eklenecek: Güneş'in bant bant mutlak kadirleri, gezegen kütle parametrelerinin tamamı [Park2021], Dünya elipsoidi (WGS84 ve GRS80) [IERS2010].

**`aci`** (uygulandı). Derece, yay saniyesi ve radyan dönüşümleri, aralık indirgeme, birim vektör dönüşümleri, açısal ayrıklık [ERFA2023]. Eklenecek:

- `altmislik_den_derece`, `derece_den_altmislik`: dizgi ayrıştırma ve biçimlendirme; eksi sıfır durumu.
- `konum_acisi(l1, b1, l2, b2) → rad`: kuzeyden doğuya.
- `hedef_nokta(l, b, konum_acisi, ayriklik) → (l, b)`.

**`vektor`**. `donme_1/2/3(aci) → 3×3`, matris çarpımı ve devriği, Rodrigues dönmesi, küresel hız dönüşümü [SOFA2023].

**`sayisal`**. `kok_bul_brent`, `gauss_legendre`, `romberg`, `rk4_adim`, `dormand_prince` [DormandPrince1980], `leapfrog_adim`, `yoshida4_adim` [Yoshida1990], `chebyshev_degerlendir` (Clenshaw), `kubik_spline`, `lanczos_cekirdegi`, `kahan_toplam` [Higham2002].

## `zaman`

**`julyen`** (uygulandı). Takvim ↔ MJD/JD, TAI ↔ TT, Dünya dönme açısı, Greenwich ve yerel ortalama yıldız zamanı, ortalama eğiklik [ERFA2023].

**`olcekler`** (kısmen uygulandı: `tai_eksi_utc_s`, `tt_eksi_utc_s` [ERFA2023]).

- `utc_den_tai(jd1, jd2) → (jd1, jd2)`: artık saniye gününün ele alınmasıyla [IERS2010].
- `tt_den_tdb(jd1, jd2, ...) → (jd1, jd2)`: tek terimli düşük duyarlık biçimi ve tam seri [Kaplan2005].
- `utc_den_ut1(jd1, jd2, dut1_s)`: DUT1 dışarıdan verilir.
- `delta_t_s(yil)`: tarihsel polinomlar [Meeus1998].

**`barisentrik`**. `romer_gecikmesi_s`, `bjd_tdb` [Eastman2010]; `barisentrik_hiz_duzeltmesi_m_s` [WrightEastman2014].

**`olaylar`**. `dogus_batis_meridyen`, `alacakaranlik`, `ay_evresi` [Meeus1998].

## `koordinat`

**`donusum`** (uygulandı). ICRS ↔ galaktik, saat açısı ↔ ufuk, düzlem-paralel hava kütlesi [ERFA2023].

Eklenecek:

- `ekvatoral_den_ekliptik(ra, dec, egiklik) → (lam, bet)` ve tersi.
- `presesyon_matrisi_iau2006(tt) → 3×3` [Capitaine2003]; `nutasyon_iau2000b(tt) → (dpsi, deps)` [IERS2010].
- `ozhareket_uygula(ra, dec, pmra, pmdec, paralaks, vr, devir1, devir2)` [SOFA2023].
- `yillik_aberasyon`, `isik_bukulmesi`, `gunluk_aberasyon` [Kaplan2005].
- `kirilma_bennett(yukseklik, basinc, sicaklik) → rad` [Bennett1982].
- `hava_kutlesi_kasten_young(zenit) → X` [KastenYoung1989], `hava_kutlesi_young(zenit)` [Young1994].
- `icrs_den_gozlenen(...)`: zincirin tamamı; ERFA `atco13` ile karşılaştırılır.

**`izdusum`**. `tan_izdusum`, `tan_ters`, `sin`, `stg`, `arc`, `zea`, `car`, `ait` [CalabrettaGreisen2002]; `piksel_den_gok`, `gok_den_piksel` (`CRPIX`, `CD`, SIP) [GreisenCalabretta2002; Shupe2005].

**`healpix`**. `aci_den_piksel`, `piksel_den_aci`, `komsular`, halka ↔ iç içe [Gorski2005].

## `efemeris`

**`kepler`** (uygulandı). Kepler denklemi (eliptik), gerçek anomali, dönem, vis-viva; kaynak kayıtları okunmayı bekliyor [MurrayDermott1999].

Eklenecek:

- `kepler_markley(M, e) → E`: yinelemesiz çözüm [Markley1995].
- `hiperbolik_anomali`, `evrensel_degisken_ilerlet` [Vallado2022].
- `elemanlar_dan_durum(a, e, i, Ω, ω, M, μ) → (r, v)` ve tersi [Vallado2022].
- `gunes_konumu_dusuk`, `ay_konumu_dusuk`, `gezegen_konumu_dusuk` [Meeus1998].
- `spk_oku`, `spk_konum_hiz`: DE440 [Park2021].
- `n_cisim_ilerlet(durum, kutleler, dt, adim, yontem)`: leapfrog, Wisdom-Holman [WisdomHolman1991; ReinTamayo2015].
- `gecis_isik_egrisi(t, parametreler)`: kenar kararmalı [MandelAgol2002].
- `dikine_hiz_egrisi(t, K, P, e, ω, T0, γ)`.

## `isinim`

**`kadir`** (uygulandı). Bolometrik kadir ↔ ışınım gücü [Mamajek2015], uzaklık modülü [Hogg1999], Planck fonksiyonu (kaynak kaydı bekliyor) [RybickiLightman1979], kara cisim akısı.

Eklenecek:

- `ab_kadiri(aki_jy)`, `aki_jy(ab_kadiri)` [OkeGunn1983].
- `asinh_kadir(aki, yumusatma)` [Lupton1999].
- `filtre_oku`, `sentetik_kadir(tayf, filtre, sistem)` [BessellMurphy2012].
- `pivot_dalgaboyu`, `etkin_dalgaboyu`.
- `sonumleme_ccm89(dalga_boyu, rv)`, `sonumleme_f99` [CCM1989; Fitzpatrick1999].
- `wesenheit(kadir, renk, r)` [Madore1982].
- `k_duzeltmesi(tayf, z, filtre_goz, filtre_durgun)` [Hogg2002].

## `gozlem`

**`dedektor`** (uygulandı; kaynak kayıtları bekliyor). `ham_kare_uret`, `yildiz_alani`, `sinyal_gurultu_orani` [Howell2006; MerlineHowell1995].

Eklenecek:

- `hedef_gorunurlugu(hedef, gozlemci, zamanlar) → (yukseklik, hava_kutlesi)`.
- `ay_ayrikligi_ve_aydinlanma`; `gokyuzu_parlakligi_ay` [KrisciunasSchaefer1991].
- `poz_suresi(hedef_sn, ...)`, `doyma_suresi`.
- `gorus_fwhm(r0, dalga_boyu, hava_kutlesi)` [Fried1966].
- Üretece eklenecekler: kozmik ışın, sıcak piksel, doyma, Moffat PSF [Moffat1969].
- `foton_aktarim_egrisi(duz_cifti_listesi)`: çok düzeyli varyans-ortalama uyumu [Janesick2007].

## `indirgeme`

**`kalibrasyon`** (uygulandı; kaynak kayıtları bekliyor). `birlestir`, `kare_indirge`, `kazanc_ve_okuma_gurultusu` [Howell2006; Janesick2007].

Eklenecek:

- Maske düzlemi ve ana kalibrasyon karelerinin gürültüsünün varyansa eklenmesi.
- `kozmik_isin_laplace(goruntu, ...)` [vanDokkum2001].
- `arka_plan_haritasi(goruntu, hucre, suzgec)` [BertinArnouts1996].
- `hizala`, `yeniden_ornekle`, `birlestir`; `drizzle` [FruchterHook2002].
- `psf_esleme_cekirdegi`, `fark_goruntusu` [AlardLupton1998; Zackay2016].

## `fotometri` ve `astrometri`

- `yildiz_bul(goruntu, fwhm, esik)` [Stetson1987]; `bolutle`, `ayristir` [BertinArnouts1996].
- `merkez_moment`, `merkez_gauss`; `ikinci_momentler`.
- `aciklik_fotometrisi(goruntu, konumlar, r, halka)`: tam piksel-daire kesişimi [MerlineHowell1995].
- `kron_yaricapi` [Kron1980]; `en_iyi_fotometri` [Naylor1998].
- `psf_modeli_kur`, `psf_fotometrisi` [Stetson1987].
- `sifir_noktasi_uydur(aygit_kadiri, katalog_kadiri, renk, hava_kutlesi)` [Landolt1992; Padmanabhan2008].
- `katalog_esle(xy, gok, tolerans)`; `wcs_uydur(xy, radec, derece)` [Lang2010; Shupe2005].

## `tayf`

- `iz_bul`, `gokyuzu_cikar`, `en_iyi_cikarim` [Horne1986].
- `ark_cizgileri_bul`, `dalgaboyu_cozumu`; `hava_den_bosluk` [Ciddor1996].
- `duyarlilik_fonksiyonu`, `aki_kalibre`.
- `sureklilik_normallestir`, `cizgi_uydur` (Gauss, Voigt), `esdeger_genislik`.
- `capraz_ilinti_kirmiziya_kayma(tayf, sablon)` [TonryDavis1979].

## `zamanserisi`

**`periyodogram`** (uygulandı). Klasik ve yüzen ortalamalı ağırlıklı Lomb-Scargle, yanlış alarm olasılığı, frekans ızgarası [VanderPlas2018].

Eklenecek:

- `pencere_periyodogrami`, `cok_terimli_lomb_scargle` [VanderPlas2018]; `baluev_fap` [Baluev2008].
- `faz_katla`, `kutula`; `faz_sacilimi_minimizasyonu` [Stellingwerf1978].
- `egilim_gider_medyan`, `egilim_gider_ciftagirlik`.
- `kutu_en_kucuk_kareler(t, aki, donemler, sureler)` [Kovacs2002]; `gecis_en_kucuk_kareler` [HippkeHeller2019].
- `gp_log_olabilirlik(cekirdek, t, y, sigma)` [RasmussenWilliams2006].

## `istatistik`

**`uyum`** (uygulandı). Kovaryanslı χ², doğrusal en küçük kareler, doğru uydurma [HoggBovyLang2010].

**`mcmc`** (uygulandı). Gerdirme hamlesi, topluluk örnekleyici [ForemanMackey2013], bölünmüş ve sıra normalleştirilmiş R̂ [Vehtari2021], öz ilinti süresi (kaynak kaydı bekliyor).

**`saglam`** (uygulandı; kaynak kayıtları bekliyor). Sigma kırpma, MAD ölçeği, çift ağırlıklı konum ve ölçek [Beers1990].

Eklenecek:

- `yayilim_dogrusal(f, x, kov)`, `yayilim_monte_carlo`.
- `dogru_uydur_iki_eksen(x, y, sx, sy, ic_sacilma)` [HoggBovyLang2010; Kelly2007].
- `karisim_modeli_aykiri` [HoggBovyLang2010].
- `metropolis_hastings` [Metropolis1953; Hastings1970]; etkin örnek sayısı [Vehtari2021]; klasik R̂ ile karşılaştırma [GelmanRubin1992].
- `ic_ice_ornekle` [Skilling2006].
- `bootstrap`, `jackknife` [Efron1979].
- `paralakstan_uzaklik_sonsali(paralaks, sigma, onsel)` [BailerJones2015].
- `cash_istatistigi` [Cash1979].

## `kozmoloji`

**`uzaklik`** (uygulandı). `E(z)`, eşyürüyen, enine eşyürüyen, açısal çap ve ışınım gücü uzaklıkları, uzaklık modülü, geriye bakış zamanı, hacim elemanı ve eşyürüyen hacim, iki kırmızıya kayma arası açısal çap uzaklığı, Doppler kayması ve özel hız, Hubble uzaklığı ve zamanı [Hogg1999; Brout2022].

**`hubble_diyagrami`** (uygulandı). Veri seçimi, artık vektörü, olabilirlik, kapalı biçimli en iyi mutlak kadir [Brout2022].

Eklenecek:

- Radyasyon ve kütleli nötrino terimleri; `w0wa` parametrelemesi [ChevallierPolarski2001; Linder2003].
- `evrenin_yasi_gyr` (radyasyon terimiyle birlikte).
- `kozmografik_uzaklik(z, h0, q0, j0)` [Riess2022].
- `ses_ufku_eh98(omega_m_h2, omega_b_h2)` [EisensteinHu1998]; `bao_uzakliklari(z)` (`D_V`, `D_M/r_d`, `D_H/r_d`) [DESI2025].
- `buyume_carpani(z)` [Dodelson2020].
- `tripp_uzaklik_modulu(mB, x1, c, alfa, beta, M)` [Tripp1998].

## `yildiz` ve `gokada`

- `lane_emden(n) → (ξ, θ)` [Kippenhahn2012]; `beyaz_cuce_kutle_yaricap` [ShapiroTeukolsky1983].
- `imf_salpeter`, `imf_kroupa`, `imf_chabrier`: yoğunluk ve örnekleme [Salpeter1955; Kroupa2001; Chabrier2003].
- `izokron_oku`, `izokron_enterpole` [Choi2016].
- `leavitt_uzaklik_modulu(donem, kadir, renk)` [Madore1982].
- Potansiyel arayüzü: `yogunluk`, `potansiyel`, `kuvvet`, `dairesel_hiz`; Plummer, Hernquist, NFW, Miyamoto-Nagai [Plummer1911; Hernquist1990; NFW1997].
- `sersic_profili`, `sersic_bn`, `sersic_toplam_aki` [GrahamDriver2005].
- `yorunge_ilerlet(potansiyel, durum, dt, adim)`.

## `arsiv` ve `io`

**`arsiv.pantheon`** (uygulandı). İndirme, özet kaydı, tablo ve kovaryans okuma [Scolnic2022; Brout2022].

Eklenecek: `onbellek` (genel indirme ve özet kaydı), `tap` (eşzamanlı ve eşzamansız sorgu), `gaia`, `mast`, `horizons`; `io.fits` (başlık, görüntü, ikili tablo) [Pence2010; FITS2018], `io.votable`.

## Sıra

Yol haritasındaki aşamalar bu envanterin okunma sırasını verir ([07_yol_haritasi.md](07_yol_haritasi.md)). Bir fonksiyon yazılmadan önce kaynağı açılır; kaynağa erişilemiyorsa fonksiyon `teyit="bekliyor"` ile yazılır ve [DURUM.md](DURUM.md) dosyasının bekleyenler listesine düşer.
