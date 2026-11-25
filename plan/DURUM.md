# Uygulama durumu

Bu dosya `python kaynakca/kutuk_raporu.py` ile kaynak kütüğünden üretilir; elle düzenlenmez.

- Kaynağa bağlı açık fonksiyon: 205
- Bütün kayıtları kaynak metninden okunmuş fonksiyon: 190
- Kaynak kaydı: 244 (okundu: 228, bekliyor: 16)
- Kaynaklı sabit: 35 (teyit bekleyen: 0)
- Test fonksiyonu: 263 (parametreli ve özellik testleri tek sayılır)
- Kaynak kuralından muaf altyapı fonksiyonu: 4

`okundu`: denklem kod yazılırken kaynağın metninden okundu. `bekliyor`: denklem hatırlanan biçimiyle
yazıldı, testlerle sınandı, ama kaynağın metni henüz açılmadı.

## Fonksiyonlar

### `kozmos.arsiv.gaia`

- `mutlak_kadir`: [Babusiaux2018] Bölüm 2: M_G = G + 5 + 5 log10(ϖ/1000), ϖ miliyay saniyesi (okundu)
- `yakin_yildizlar`: [Vallenari2023] Gaia DR3 gaia_source tablosu (bekliyor); [Babusiaux2018] Ek B: arşiv sorgusu ve süzgeçler; Bölüm 2 (okundu)

### `kozmos.arsiv.pantheon`

- `indir`: [Scolnic2022] veri deposu, 4_DISTANCES_AND_COVAR (okundu)
- `kovaryans_oku`: [Brout2022] veri deposu README: .cov biçimi (okundu)
- `tablo_oku`: [Brout2022] veri deposu README: sütun tanımları (okundu)
- `yukle`: [Brout2022] veri deposu README (okundu)

### `kozmos.arsiv.skyview`

- `kesit_indir`: [SkyView2026] Batch Page: runquery.pl örnekleri (Position, Survey, Pixels, Size, Return) (okundu)

### `kozmos.arsiv.tap`

- `csv_oku`: [TAP2019] Bölüm 2.7.3 (FORMAT): text/csv çıktısı (okundu)
- `sorgula`: [TAP2019] Bölüm 2.1 (TAP-sync), 2.7.1 (LANG), 2.7.2 (QUERY), 2.7.3 (FORMAT) (okundu)

### `kozmos.efemeris.kepler`

- `dis_merkezli_anomali`: [MurrayCorreia2010] TeX etiketi eq37 (Kepler denklemi), eq38-eq39 (Newton-Raphson) (okundu)
- `gercek_anomali`: [MurrayCorreia2010] TeX etiketi eq17 ve eq33 (okundu)
- `yorunge_donemi_s`: [MurrayCorreia2010] TeX etiketi eq19 (okundu)
- `yorunge_hizi_m_s`: [MurrayCorreia2010] TeX etiketi eq30 (okundu)

### `kozmos.fotometri.kaynak`

- `aciklik_fotometrisi`: [Howell2006] Bölüm 5: açıklık fotometrisi (bekliyor)
- `arka_plan_haritasi`: [SExtractorBelge2026] Modeling the background: Background map (okundu)
- `arka_plan_kestir`: [SExtractorBelge2026] Modeling the background: Background estimation, Denk. 4 (okundu)
- `kaynak_bul`: [BertinArnouts1996] saptama: eşik üstü bağlı pikseller (bekliyor)

### `kozmos.gorsel.germe`

- `asinh_ger`: [Lupton2004] Bölüm 2, dipnot: f(x) = asinh(α Q (x − m)) / Q (okundu)
- `lupton_rgb`: [Lupton2004] Bölüm 2, TeX etiketi EqIntensityStretch (okundu)
- `yuzdelik_olcekle`: [Lupton2004] Bölüm 2: doğrusal eşleme ve en büyük değer M (okundu)

### `kozmos.gozlem.dedektor`

- `ham_kare_uret`: [CraigChambers2026] defter 01-05 'Calibration overview': ham görüntü denklemi (okundu)
- `sinyal_gurultu_orani`: [MerlineHowell1995] arka plan kestirimi terimi (1 + n_pix/n_B) (bekliyor); [Howell2006] Bölüm 4.4: CCD denklemi (bekliyor)
- `yildiz_alani`: [Howell2006] Bölüm 5: nokta dağılım fonksiyonu (dairesel Gauss yaklaşımı) (bekliyor)

### `kozmos.gozlem.planlama`

- `gorunurluk`: [ERFA2023] gmst06.c, hd2ae.c (okundu)
- `meridyen_gecisi_jd`: [ERFA2023] era00.c (dönme hızı), hd2ae.c (okundu)
- `ufuk_ustu_suresi_saat`: [ERFA2023] hd2ae.c (yükseklik bileşeni) (okundu)

### `kozmos.indirgeme.kalibrasyon`

- `birlestir`: [CraigChambers2026] defter 01-05 'Calibration overview': 'Every image has noise' (okundu)
- `kare_indirge`: [CraigChambers2026] defter 01-05 'Calibration overview': indirgeme denklemi (okundu)
- `kazanc_ve_okuma_gurultusu`: [IRAF2026] noao/obsutil/src/doc/findgain.hlp, ALGORITHM (okundu)

### `kozmos.io.fits`

- `baslik_coz`: [FITS2018] Bölüm 3.3.1 ve 4.1: başlık blokları, anahtar kaydı bileşenleri; END anahtarı (okundu)
- `fits_oku`: [FITS2018] Tablo 8 (BITPIX), Denk. 3 (BZERO, BSCALE), Bölüm 5 (büyük uçlu bayt sırası) (okundu)
- `fits_yaz`: [FITS2018] Bölüm 3.3.1, 4.4.1 (zorunlu anahtarlar), Tablo 8, Bölüm 5 (okundu)

### `kozmos.isinim.kadir`

- `bolometrik_kadirden_isinim_gucu_w`: [Mamajek2015] M_Bol tanımı; L_0 = 3,0128e28 W (okundu)
- `kara_cisim_akisi`: [CODATA2022] Stefan-Boltzmann constant (okundu)
- `mutlak_bolometrik_kadir`: [Mamajek2015] M_Bol tanımı; L_0 = 3,0128e28 W (okundu)
- `planck_dalgaboyu`: [Astropy2022] modeling/physical_models.py, BlackBody.evaluate (okundu)
- `planck_frekans`: [Astropy2022] modeling/physical_models.py, BlackBody.evaluate (okundu)
- `uzaklik_modulu_pc`: [Hogg1999] Denk. 25 (okundu)
- `uzaklik_modulunden_pc`: [Hogg1999] Denk. 25 (okundu)

### `kozmos.isinim.kizilotesi`

- `akidan_vega_kadiri`: [Wright2010] Bölüm 4.4: F_ν° (okundu); [Cohen2003] Tablo 2: F_ν(iso) (okundu)
- `tepe_dalgaboyu_um`: [CODATA2022] Wien wavelength displacement law constant (okundu)
- `tepe_sicakligi_k`: [CODATA2022] Wien wavelength displacement law constant (okundu)
- `vega_ab_farki`: [Wright2010] Bölüm 4.4: F_ν° (okundu); [Cohen2003] Tablo 2: F_ν(iso) (okundu)
- `vega_kadirinden_aki_jy`: [Wright2010] Bölüm 4.4: F_ν° (okundu); [Cohen2003] Tablo 2: F_ν(iso) (okundu)

### `kozmos.istatistik.mcmc`

- `bolunmus_rhat`: [Vehtari2021] Bölüm 3.1: B, W, var⁺ ve R̂ denklemleri (okundu)
- `gerdirme_carpani`: [ForemanMackey2013] Denk. goodmanprop (g(z) ∝ 1/√z, z ∈ [1/a, a]) (okundu)
- `oz_ilinti_suresi`: [ForemanMackey2013] Bölüm 3, 'Autocorrelation time': τ_f ve C_f(T) denklemleri (okundu)
- `rhat_sira_normallestirilmis`: [Vehtari2021] Bölüm 4.1-4.2: sıra normalleştirme ve katlama denklemleri (okundu)
- `topluluk_ornekle`: [ForemanMackey2013] Denk. proposal, acceptance; Algoritma 3 (paralel gerdirme hamlesi) (okundu)

### `kozmos.istatistik.saglam`

- `ciftagirlik_konum`: [Astropy2022] stats/biweight.py, biweight_location belge dizgisi (okundu)
- `ciftagirlik_olcek`: [Astropy2022] stats/biweight.py, biweight_scale belge dizgisi (okundu)
- `mad_olcek`: [Astropy2022] stats/funcs.py, median_absolute_deviation ve mad_std (okundu)
- `sigma_kirp`: [Astropy2022] stats/sigma_clipping.py, SigmaClip belge dizgisi (okundu)

### `kozmos.istatistik.uyum`

- `dogru_uydur`: [HoggBovyLang2010] TeX etiketi eq:fofx, eq:lsf (okundu)
- `dogrusal_en_kucuk_kareler`: [HoggBovyLang2010] TeX etiketi eq:lsf ve parametre kovaryansı denklemi (okundu)
- `ki_kare`: [HoggBovyLang2010] TeX etiketi eq:chisquared (okundu)

### `kozmos.karadelik.temel`

- `bekenstein_hawking_entropisi`: [Page2005] TeX etiketi eq:2, S_bh = S_BH ≡ A/4 (Planck birimleri); A = 16π M² (Schwarzschild, eq:3 öncesi metin) (okundu)
- `birlesmeye_kalan_sure_s`: [Sathyaprakash2009] §'Coalescing binaries' paragrafı, Newton mertebesi cıvıltı süresi τ0 (etiketsiz denklem) (okundu)
- `cirp_kutlesi_kg`: [Sathyaprakash2009] TeX etiketi eqn:chirpmassdef, 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M (okundu)
- `ergosfer_yaricapi_kerr_rg`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', r_0 (ergosfer; aynı eqnarray, etiketsiz satır) (okundu)
- `foton_kuresi_yaricapi_m`: [Abramowicz2013] 'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), x_ph = 3 (okundu)
- `foton_yorungesi_yaricapi_kerr_rg`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', TeX etiketi photon-radius-kerr (okundu)
- `golge_acisal_capi_uas`: [EHT2019] §1 R_c = √27 r_g; §7 'Model Comparison and Parameter Estimation' θ_g ≡ GM/(c² D), d = α θ_g (okundu)
- `gw_frekans_turevi_hz_s`: [Sathyaprakash2009] TeX etiketi eqn:circularPM (Ṗ_b, Peters-Mathews 1963'e atfen), f_gw = 2/P_b (okundu)
- `gw_frekansi_isco_hz`: [Sathyaprakash2009] paragraf 'Waveform polarizations.' sonrası metin: ışınım en düşük mertebede yörünge frekansının iki katında yayılır (okundu); [Abramowicz2013] 'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), Ω_K² = GM/r³, x_ms = 6 (okundu)
- `hawking_buharlasma_suresi_s`: [Page2005] TeX etiketi eq:3 (A, κ) ve §1 (T = κ/2π); yalnızca T_H ve A girdisi (okundu); [Page1976] Hawking ışınımıyla kütle kaybı ve ömür (greybody düzeltmeli kesin sonuç orada) (bekliyor)
- `hawking_sicakligi_k`: [Page2005] §1; T = κ/(2π) ve TeX etiketi eq:3 (Schwarzschild: κ = 1/(4M), Planck birimlerinde) (okundu); [Abramowicz2013] §1 (Giriş), Hawking sıcaklığı dipnotu: T_H = ħc³/(8π G M k_B) (okundu)
- `isco_isinim_verimliligi`: [Bardeen1972] ISCO'daki özgül enerji Ẽ_ms (bekliyor); [Abramowicz2013] §1 (Giriş), verim tanımı η ≡ L/(Ṁ c²) (okundu)
- `isco_ozgul_enerji_kerr`: [Bardeen1972] ISCO'daki özgül enerji Ẽ_ms = (1 − 2/(3 r̃_ms))^{1/2} (bekliyor)
- `isco_yaricapi_kerr_rg`: [Bardeen1972] ISCO formülü r_ms = M{3 + Z2 ∓ [(3 − Z1)(3 + Z1 + 2Z2)]^{1/2}} (retrograde için + işareti) (bekliyor); [Abramowicz2013] 'Summary: characteristic radii and frequencies', TeX etiketi stable-radius-kerr (Bardeen vd. formülü) (okundu)
- `isco_yaricapi_schwarzschild_m`: [Abramowicz2013] 'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), x_ms = 6 (okundu)
- `isco_yorunge_frekansi_hz`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', Ω_K² = GM/r³ (TeX etiketi schwarzschild-frequencies), x_ms = 6 (okundu)
- `kalan_sureden_gw_frekansi_hz`: [Sathyaprakash2009] §'Coalescing binaries' paragrafı, τ0 (etiketsiz denklem); tersi tarafımdan alındı (okundu)
- `kritik_carpisma_parametresi_m`: [EHT2019] §1 (Giriş), 'foton yakalama yarıçapı' R_c = √27 r_g, sayfa L1 (okundu)
- `kutleceken_yaricapi_m`: [EHT2019] §1 (Giriş), r_g ≡ GM/c² tanımı, sayfa L1 (okundu)
- `schwarzschild_yaricapi_m`: [EHT2019] §1 (Giriş), R_S = 2 r_g, sayfa L1 (okundu)
- `ufuk_acisal_hizi_rad_s`: [Page2005] TeX etiketi eq:3, Ω = 4π a / A (Q = 0) (okundu)
- `ufuk_yaricapi_kerr_rg`: [Page2005] TeX etiketi eq:3, r_+ (Q = 0) (okundu); [Abramowicz2013] 'Summary: characteristic radii and frequencies', r_H (aynı eqnarray, etiketsiz satır) (okundu)
- `yercekimsel_kirmiziya_kayma`: [Carroll2019] Schwarzschild metriği, g_tt = −(1 − 2GM/rc²); statik gözlemci (bekliyor)

### `kozmos.karadelik.yigilma`

- `bondi_kutle_akisi_kg_s`: [Inayoshi2020] Tablo 3, Ṁ_B ≡ π e^{3/2} ρ G² M² / c_s³ (okundu)
- `bondi_yaricapi_m`: [Inayoshi2020] §'Inflow from large scales' (R_B ≡ G M / c_s²) ve Tablo 3 (TeX etiketi tab:phys) (okundu)
- `eddington_isinim_gucu_w`: [Inayoshi2020] §'Growing BHs by Accretion: Is There an Eddington Limit?', L_Edd = 4π c G M / κ_es (okundu)
- `eddington_kutle_akisi_kg_s`: [Inayoshi2020] §'Growing BHs by Accretion' ve Tablo 3, Ṁ_Edd ≡ L_Edd / (0,1 c²) (dipnotta L_Edd/c² tanımına da değinilir) (okundu)
- `gelgit_parcalanma_yaricapi_m`: [Komossa2015] §1 (Giriş), gelgit yarıçapı r_t denklemi (etiketsiz): r_t ≃ 7e12 (M/10⁶ M☉)^{1/3} (m*/M☉)^{-1/3} (r*/R☉) cm (okundu)
- `hills_kutlesi_kg`: [Komossa2015] §1, r_t denklemi ve devamındaki cümle: Güneş türü yıldızlar ≳ 10⁸ M☉'de bütün yutulur, beyaz cüceler ≲ 10⁵ M☉'de parçalanır (okundu)
- `ince_disk_azami_sicakligi_k`: [Armitage2011] §2 'Accretion Disk Evolution', T_eff profili; azami değer tarafımdan türetildi (okundu)
- `ince_disk_azami_sicaklik_yaricapi_m`: [Armitage2011] §2 'Accretion Disk Evolution', T_eff profili; azami konum tarafımdan türetildi (okundu)
- `ince_disk_etkin_sicakligi_k`: [Armitage2011] §2 'Accretion Disk Evolution', T_eff⁴ = 3 G M Ṁ / (8π σ r³) [1 − √(r_in/r)] (TeX etiketi eq:2_nusigma ile aynı eqnarray, ikinci satır) (okundu)
- `salpeter_zamani_s`: [Inayoshi2020] §'Growing BHs by Accretion', t_grow ≈ 0,45 ε/(1−ε) f_duty^{-1} ln(M/M_tohum) Gyr; §1 t_Edd ≈ 50 Myr (ε = 0,1) (okundu)
- `yigilma_isinim_gucu_w`: [Abramowicz2013] §1 (Giriş), verim tanımı η ≡ L/(Ṁ c²) (okundu)

### `kozmos.koordinat.donusum`

- `galaktik_den_icrs`: [ERFA2023] icrs2g.c (okundu)
- `hava_kutlesi_duzlem_paralel`: [ERFA2023] hd2ae.c (z bileşeni) (okundu)
- `icrs_den_galaktik`: [ERFA2023] icrs2g.c (okundu)
- `saat_acisi_den_ufuk`: [ERFA2023] hd2ae.c (okundu)
- `ufuk_den_saat_acisi`: [ERFA2023] ae2hd.c (okundu)

### `kozmos.kozmoloji.hubble_diyagrami`

- `artik_vektoru`: [Brout2022] eq:dmu, eq:dmuprime, eq:dl (okundu)
- `en_iyi_mutlak_kadir`: [Brout2022] eq:likelihood (okundu)
- `log_olabilirlik`: [Brout2022] eq:likelihood, eq:sh0eslikelihood (okundu)
- `veri_hazirla`: [Brout2022] Bölüm 4: z < ~0,01 Hubble diyagramı kullanılmaz; eq:dmuprime (okundu)

### `kozmos.kozmoloji.uzaklik`

- `acisal_cap_uzakligi_mpc`: [Hogg1999] Denk. 18 (okundu)
- `acisal_cap_uzakligi_z1_z2_mpc`: [Hogg1999] Denk. 19 (okundu)
- `doppler_hizi_km_s`: [Hogg1999] Denk. 9 (okundu)
- `doppler_kirmiziya_kayma`: [Hogg1999] Denk. 9 (okundu)
- `e_fonksiyonu`: [Brout2022] H(z) denklemi (TeX etiketi eq:hz) (okundu); [Hogg1999] Denk. 14 (okundu)
- `enine_esyuruyen_uzaklik_mpc`: [Hogg1999] Denk. 16 (okundu)
- `esyuruyen_hacim_elemani_mpc3`: [Hogg1999] Denk. 28 (okundu)
- `esyuruyen_hacim_mpc3`: [Hogg1999] Denk. 29 (okundu)
- `esyuruyen_uzaklik_mpc`: [Hogg1999] Denk. 15 (okundu)
- `geriye_bakis_zamani_gyr`: [Hogg1999] Denk. 30 (okundu)
- `hubble_uzakligi_mpc`: [Hogg1999] Denk. 4 (okundu)
- `hubble_zamani_gyr`: [Hogg1999] Denk. 3 (okundu)
- `isinim_gucu_uzakligi_mpc`: [Hogg1999] Denk. 21 (okundu)
- `ozel_hiz_km_s`: [Hogg1999] Denk. 10 (okundu)
- `uzaklik_modulu`: [Hogg1999] Denk. 25 (okundu)

### `kozmos.temel.aci`

- `acisal_ayriklik`: [ERFA2023] sepp.c (okundu)
- `birim_vektor_den_kuresel`: [ERFA2023] c2s.c (okundu)
- `derece_den_radyan`: [ERFA2023] erfam.h ERFA_DD2R (okundu)
- `eksi_pi_arti_pi`: [ERFA2023] anpm.c (okundu)
- `kuresel_den_birim_vektor`: [ERFA2023] s2c.c (okundu)
- `radyan_dan_derece`: [ERFA2023] erfam.h ERFA_DD2R (okundu)
- `sifir_iki_pi`: [ERFA2023] anp.c (okundu)
- `yaysaniye_den_radyan`: [ERFA2023] erfam.h ERFA_DAS2R (okundu)

### `kozmos.yildiz.imf`

- `chabrier_dn_dm`: [Chabrier2003] Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1 (table.diskIMF), 'IMF' satırı (okundu)
- `chabrier_kutle_orani_ustunde`: [Chabrier2003] Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1 (okundu)
- `chabrier_ornekle`: [Chabrier2003] Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1 (okundu)
- `chabrier_ortalama_kutle`: [Chabrier2003] Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1 (okundu)
- `kroupa_dn_dm`: [Kroupa2001] Sec. 'The universal IMF', TeX etiketleri eq:xi ve eq:imf (okundu)
- `kroupa_kutle_orani_ustunde`: [Kroupa2001] Sec. 'The universal IMF', TeX etiketi eq:imf ve kütle payları cümleleri (okundu)
- `kroupa_ornekle`: [Kroupa2001] Sec. 'The universal IMF', TeX etiketi eq:imf (okundu)
- `kroupa_ortalama_kutle`: [Kroupa2001] Sec. 'The universal IMF', TeX etiketi eq:imf ve ortalama kütle cümlesi (okundu)
- `salpeter_dn_dm`: [Salpeter1955] özgün makale (eğim değeri) (bekliyor); [Kroupa2001] Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35 (okundu); [Chabrier2003] Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1): α = 2,35, x = 1,35 (okundu)
- `salpeter_kutle_orani_ustunde`: [Salpeter1955] özgün makale (eğim değeri) (bekliyor); [Kroupa2001] Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35 (okundu); [Chabrier2003] Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1) (okundu)
- `salpeter_ornekle`: [Salpeter1955] özgün makale (eğim değeri) (bekliyor); [Kroupa2001] Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35 (okundu); [Chabrier2003] Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1) (okundu)
- `salpeter_ortalama_kutle`: [Salpeter1955] özgün makale (eğim değeri) (bekliyor); [Kroupa2001] Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35 (okundu); [Chabrier2003] Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1) (okundu)

### `kozmos.yildiz.olcekleme`

- `beyaz_cuce_k_katsayisi_si`: [Bludman2010] Sec. II.A (göreli dejenere yıldızlar): K_WD = (hc/8)(3/π)^{1/3}(m_H μ_e)^{-4/3} (okundu)
- `chandrasekhar_kutlesi_kg`: [Bludman2010] Sec. II (giriş) M = 4π 0ω_3 (K/πG)^{3/2}; Sec. II.A M_Ch = 5,83 M_⊙/μ_e² = 1,46 M_⊙ (okundu)
- `eddington_beta`: [Bludman2010] Sec. II.B, Eddington dördüncü derece denklemi: (1-β)/β⁴ = (M μ²/M_*)² (okundu)
- `eddington_isinim_gucu_w`: [Bludman2010] Sec. II.B, Eddington standart modeli: L_Edd := 4π c G M/κ_p (okundu); [Paxton2013] Sec. 'Rotationally-Enhanced Mass Loss' (TeX etiketi s.rotation-mass-loss): L_Edd = 4π c G M/κ (okundu)
- `eddington_olcek_kutlesi_kg`: [Bludman2010] Sec. II.B: M_* := 3√10 0ω_3/π³ (hc/(G m_H^{4/3}))^{3/2} = 18,3 M_⊙ (okundu)
- `etkin_sicaklik_k`: [Prsa2016] metin: T_eff, Stefan-Boltzmann yasasından; Tablo 'formu', M_Bol satırı (okundu)
- `isinim_gucu_w`: [Prsa2016] Tablo 'formu', M_Bol satırı (4πσR² terimi); metin: T_eff Stefan-Boltzmann yasasından (okundu)
- `kelvin_helmholtz_zaman_olcegi_s`: [Zinnecker2007] Sec. 'The Accretion Phase / Formation of the Hydrostatic Core' metni ve GLOSSARY: t_KH = G M²/(R L) (okundu)
- `kritik_acisal_hiz_rad_s`: [Paxton2013] denklem öncesi metin (mdotomega etiketi): Ω_crit² = (1 - L/L_Edd) G M / R³ (okundu)
- `log_yuzey_cekimi_cgs`: [Prsa2016] Tablo 'formu', 'log of surface gravity' satırı: 4,438068... + log M - 2 log R (okundu)
- `nukleer_zaman_olcegi_yil`: [Kippenhahn2012] nükleer zaman ölçeği tanımı (bölüm ve denklem numarasına bakılmadı) (bekliyor)
- `ortalama_yogunluk_kg_m3`: [Bludman2010] Sec. I, Tablo I öncesi metin: ρ̄_n = 3M/4πR³ (okundu)
- `serbest_dusme_zamani_s`: [McKee2007] TeX etiketi eq:tff (okundu)
- `standart_model_isinim_gucu_w`: [Bludman2010] Sec. II.B: L = L_Edd [1 - β(M)] (okundu)
- `yuzey_cekimi_m_s2`: [Prsa2016] Tablo 'formu', 'log of surface gravity' satırı: log g = log(10^6 GM) - 2 log(100 R) [cgs] (okundu)

### `kozmos.yildiz.yapi`

- `lane_emden_analitik`: [Bludman2010] Tablo II (n = 0, 1, 5 satırları) (okundu)
- `lane_emden_coz`: [Bludman2010] Tablo II (Taylor serisi 1 - ξ²/6 + n ξ⁴/120 ...) (okundu); [Liu1995] Denk. 2, 4, 5 (Lane-Emden denklemi, merkez koşulları, kuvvet serisi) (okundu)
- `lane_emden_ilk_sifir`: [Liu1995] Denk. 2 (θ'nın ilk sıfırı yıldızın yüzeyidir) (okundu); [Bludman2010] Tablo I, ξ_1n sütunu (okundu)
- `lane_emden_kutle_katsayisi`: [Bludman2010] Tablo I (0ω_3 = 2,018) ve Sec. II, M = 4π 0ω_3 (K/πG)^{3/2} (okundu); [Liu1995] Denk. 38 (m = 4π ρ_c r³ (-θ'/ξ)) (okundu)
- `lane_emden_merkez_ortalama_yogunluk_orani`: [Liu1995] Denk. 38 (r = R'de m = M) (okundu); [Bludman2010] Tablo I, ρ_cn/ρ̄_n sütunu; ρ̄_n = 3M/4πR³ tanımı (okundu)
- `politrop_k_katsayisi_si`: [Liu1995] Denk. 3 (A² tanımı, A = ξ_1/R) (okundu)
- `politrop_merkez_basinci_pa`: [Liu1995] Denk. 41 (P = K ρ_c^{(n+1)/n} ω^{n+1}, merkezde ω = 1) ve Denk. 3 (okundu)
- `politrop_merkez_sicakligi_k`: [Liu1995] Denk. 42 (T = μ m_p K ρ_c^{1/n} ω / k, merkezde ω = 1) (okundu)
- `politrop_merkez_yogunlugu_kg_m3`: [Liu1995] Denk. 38 (r = R'de), ρ̄ tanımı için [Bludman2010] Tablo I (okundu)
- `politrop_yaricap_m`: [Bludman2010] Tablo I, son sütun (M ~ R^{(n-3)/(n-1)}) (okundu); [Liu1995] Denk. 3 ve Denk. 38 (okundu)
- `politrop_yercekimi_enerjisi_j`: [Liu1995] Denk. 44 ve 45 (E_g; yüzeyde Φ = 0 ve P = 0) (okundu)
- `viriyal_ortalama_sicaklik_k`: [Liu1995] Denk. 44 (E_g = -3/(5-n) G M²/R) (okundu); [McKee2007] TeX etiketleri eq:iddot, eq:calt (viriyal teoremi) (okundu)

### `kozmos.yildizolusumu.bulut`

- `bonnor_ebert_kritik_katsayilar`: [Krumholz2015] problems/probsol2.tex, Bonnor-Ebert küresi: ODE ve (d) şıkkı (m_max = 1,18, ρ_c/ρ_s = 14,0) (okundu)
- `bonnor_ebert_kritik_kutle_kg`: [Krumholz2015] problems/probsol2.tex, Bonnor-Ebert küresi alıştırmasının (d) şıkkı (okundu); [McKee2007] TeX etiketi MBE_def (okundu)
- `bonnor_ebert_yaricapi_m`: [McKee2007] TeX etiketi MBE_def, sonraki paragraf: R_BE = 0,486 c_s / sqrt(G ρ_0) (okundu)
- `bulut_yildiz_olusum_hizi_kg_s`: [McKee2007] TeX etiketi eq:epsff (okundu)
- `dogrusal_hiz_boyut_km_s`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Scaling Relations: Larson'un ikinci yasası (okundu); [McKee2007] TeX etiketi GMC_lws (Heyer ve Brunt 2004 PCA uyumu) (okundu)
- `gecis_zamani_s`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Molecular Cloud Timescales: t_cr = R/σ (okundu)
- `izotermal_cokus_birikim_hizi_kg_s`: [McKee2007] TeX etiketi eq:mdssis (Shu genişleme dalgası çözümü) (okundu)
- `jeans_boyu_m`: [Krumholz2015] Bölüm 6 (Gravitational Instability and Collapse), λ_J = 2π/k_J (okundu); [McKee2007] TeX etiketi LJeans (okundu)
- `jeans_kutlesi_kg`: [Krumholz2015] Bölüm 6, Jeans kütlesi tanımı ve dipnotu (okundu); [McKee2007] TeX etiketi LJeans, ardından gelen paragraf (M_J = 2,47 M_BE) (okundu)
- `kutle_akisi_orani_normlu`: [McKee2007] TeX etiketi eq:mphi sonrası: G^{1/2}(Σ/B)_cr = 1/(2π) (sonsuz soğuk levha) (okundu)
- `kutle_yogunlugu_kg_m3`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H) (okundu)
- `mach_sayisi`: [Krumholz2015] Bölüm 4 (Gas Flows and Turbulence), Reynolds ve Mach sayıları alt bölümü: M ~ V/c_s (okundu)
- `manyetik_kritik_kutle_kg`: [Krumholz2015] Bölüm 6, Magnetic Support and the Magnetic Critical Mass (okundu); [McKee2007] TeX etiketi eq:mphi (okundu)
- `sayi_yogunlugu_m3`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H) (okundu)
- `serbest_dusme_verimliligi`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Depletion Time: ε_ff = t_ff / t_dep (okundu); [McKee2007] TeX etiketi eq:epsff (okundu)
- `serbest_dusme_zamani_s`: [Krumholz2015] Bölüm 6, Jeans analizinden sonra t_ff tanımı; Bölüm 8 zaman ölçekleri (okundu); [McKee2007] TeX etiketi eq:tff (okundu)
- `ses_hizi_m_s`: [Krumholz2015] Bölüm 4 (Gas Flows and Turbulence), Reynolds ve Mach sayıları alt bölümündeki c_s ifadesi (okundu)
- `sutun_yogunlugu_m2`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²) (okundu)
- `viryal_parametresi`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Larson'un üçüncü yasası: α_vir = 5σ²R/(GM) (okundu); [McKee2007] TeX etiketi alpha_vir_def (okundu)
- `viryal_parametresi_yuzeyden`: [McKee2007] TeX etiketi alpha_vir_def (okundu)
- `yuzey_yogunlugu_kg_m2`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²) (okundu)

### `kozmos.yildizolusumu.yasalar`

- `iki_boyutlu_jeans_kutlesi_gunes_kutlesi`: [McKee2007] TeX etiketi MJ2D (MJ2Ddef) (okundu)
- `kennicutt_schmidt_sfr_yuzey_yogunlugu`: [McKee2007] TeX etiketi eq:ks (okundu); [Kennicutt1998] Denk. 4 (metindeki sıra; 'equation (4)' olarak anılır) (okundu)
- `kennicutt_yorunge_sfr_yuzey_yogunlugu`: [McKee2007] TeX etiketi eq:ks (okundu); [Kennicutt1998] Denk. 7 (metindeki sıra; 'equation (7)' olarak anılır) (okundu)
- `serbest_dusme_sfr_yuzey_yogunlugu`: [McKee2007] TeX etiketi SFR-ff (okundu)
- `sfr_kalibrasyonu_gunes_kutlesi_yil`: [Kennicutt2012] Tablo 1 (TeX etiketi sfrequations) ve 'An Updated Compendium of Integrated SFR Calibrations' alt bölümündeki log SFR = log L_x − log C_x denklemi (okundu)
- `toomre_q_gaz`: [McKee2007] TeX etiketi Qdef (okundu)
- `tuketme_zamani_yil`: [Krumholz2015] Bölüm 8 (Giant Molecular Clouds), Depletion Time: t_dep = Σ_gaz / Σ̇_* (okundu)
- `yasam_dongusu_sfr_yuzey_yogunlugu`: [McKee2007] TeX etiketi SFR-lc (okundu)

### `kozmos.zaman.julyen`

- `dunya_donme_acisi`: [ERFA2023] era00.c (okundu)
- `greenwich_ortalama_yildiz_zamani`: [ERFA2023] gmst06.c (okundu)
- `jd_den_takvim`: [ERFA2023] jd2cal.c (okundu)
- `ortalama_egiklik`: [ERFA2023] obl06.c (okundu)
- `tai_den_tt`: [ERFA2023] erfam.h ERFA_TTMTAI, ERFA_DAYSEC (okundu)
- `takvim_den_jd`: [ERFA2023] cal2jd.c; erfam.h ERFA_DJM0 (okundu)
- `takvim_den_mjd`: [ERFA2023] cal2jd.c (okundu)
- `tt_den_tai`: [ERFA2023] erfam.h ERFA_TTMTAI, ERFA_DAYSEC (okundu)
- `yerel_ortalama_yildiz_zamani`: [Astropy2022] time/core.py, Time._sid_time_or_earth_rot_ang (okundu); [ERFA2023] gmst06.c (okundu)

### `kozmos.zaman.olcekler`

- `tai_eksi_utc_s`: [ERFA2023] dat.c (okundu)
- `tt_eksi_utc_s`: [ERFA2023] dat.c; erfam.h ERFA_TTMTAI (okundu)

### `kozmos.zamanserisi.periyodogram`

- `frekans_izgarasi`: [VanderPlas2018] Bölüm 7.1, TeX etiketi eq:n-eval (okundu)
- `lomb_scargle`: [VanderPlas2018] eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized (okundu)
- `lomb_scargle_klasik`: [VanderPlas2018] eq:lomb-scargle-periodogram, eq:tau-def (okundu)
- `yanlis_alarm_olasiligi`: [Astropy2022] timeseries/periodograms/lombscargle/_statistics.py, fap_single ve fap_naive (okundu); [VanderPlas2018] TeX etiketi eq:FAP-neff ve izleyen paragraf (okundu)

## Sabitler

- `ISIK_HIZI` = 299792458 m s^-1 (kesin) [CODATA2022] speed of light in vacuum (okundu)
- `PLANCK` = 6.62607015e-34 J Hz^-1 (kesin) [CODATA2022] Planck constant (okundu)
- `BOLTZMANN` = 1.380649e-23 J K^-1 (kesin) [CODATA2022] Boltzmann constant (okundu)
- `CEKIM` = 6.6743e-11 m^3 kg^-1 s^-2 (± 1.5e-15) [CODATA2022] Newtonian constant of gravitation (okundu)
- `TEMEL_YUK` = 1.602176634e-19 C (kesin) [CODATA2022] elementary charge (okundu)
- `ELEKTRON_KUTLESI` = 9.1093837139e-31 kg (± 2.8e-40) [CODATA2022] electron mass (okundu)
- `PROTON_KUTLESI` = 1.67262192595e-27 kg (± 5.2e-37) [CODATA2022] proton mass (okundu)
- `ATOMIK_KUTLE_BIRIMI` = 1.66053906892e-27 kg (± 5.2e-37) [CODATA2022] atomic mass constant (okundu)
- `AVOGADRO` = 6.02214076e+23 mol^-1 (kesin) [CODATA2022] Avogadro constant (okundu)
- `THOMSON_KESITI` = 6.6524587051e-29 m^2 (± 6.2e-38) [CODATA2022] Thomson cross section (okundu)
- `STEFAN_BOLTZMANN` = 5.67037441918e-08 W m^-2 K^-4 (kesin) [CODATA2022] Stefan-Boltzmann constant (5.670 374 419... e-8) (okundu; türetim: 2 pi^5 k^4 / (15 h^3 c^2))
- `WIEN_DALGABOYU` = 0.002897771955 m K (kesin) [CODATA2022] Wien wavelength displacement law constant (2.897 771 955... e-3) (okundu)
- `IKINCI_ISINIM_SABITI` = 0.014387768775 m K (kesin) [CODATA2022] second radiation constant (1.438 776 877... e-2) (okundu; türetim: h c / k)
- `GUNES_YARICAPI` = 695700000 m (kesin) [Prsa2016] Tablo 1 (okundu)
- `GUNES_ISINIM_GUCU` = 3.828e+26 W (kesin) [Prsa2016] Tablo 1 (okundu)
- `GUNES_ETKIN_SICAKLIGI` = 5772 K (kesin) [Prsa2016] Tablo 1 (okundu)
- `GUNES_SABITI` = 1361 W m^-2 (kesin) [Prsa2016] Tablo 1 (okundu)
- `GM_GUNES` = 1.3271244e+20 m^3 s^-2 (kesin) [Prsa2016] Tablo 1 (okundu)
- `GM_YER` = 3.986004e+14 m^3 s^-2 (kesin) [Prsa2016] Tablo 1 (okundu)
- `GM_JUPITER` = 1.2668653e+17 m^3 s^-2 (kesin) [Prsa2016] Tablo 1 (okundu)
- `YER_EKVATOR_YARICAPI` = 6378100 m (kesin) [Prsa2016] Tablo 1 (okundu)
- `YER_KUTUP_YARICAPI` = 6356800 m (kesin) [Prsa2016] Tablo 1 (okundu)
- `JUPITER_EKVATOR_YARICAPI` = 71492000 m (kesin) [Prsa2016] Tablo 1 (okundu)
- `JUPITER_KUTUP_YARICAPI` = 66854000 m (kesin) [Prsa2016] Tablo 1 (okundu)
- `BOLOMETRIK_SIFIR_ISINIM_GUCU` = 3.0128e+28 W (kesin) [Mamajek2015] L_0 tanımı (okundu)
- `ASTRONOMIK_BIRIM` = 149597870700 m (kesin) [ERFA2023] erfam.h ERFA_DAU (okundu)
- `GUN_SANIYE` = 86400 s (kesin) [ERFA2023] erfam.h ERFA_DAYSEC (okundu)
- `JULYEN_YILI_GUN` = 365.25 gün (kesin) [ERFA2023] erfam.h ERFA_DJY (okundu)
- `JULYEN_YUZYILI_GUN` = 36525 gün (kesin) [ERFA2023] erfam.h ERFA_DJC (okundu)
- `J2000_JD` = 2451545 gün (kesin) [ERFA2023] erfam.h ERFA_DJ00 (okundu)
- `MJD_SIFIR_JD` = 2400000.5 gün (kesin) [ERFA2023] erfam.h ERFA_DJM0 (okundu)
- `TT_EKSI_TAI` = 32.184 s (kesin) [ERFA2023] erfam.h ERFA_TTMTAI (okundu)
- `PARSEK` = 3.08567758149e+16 m (kesin) [Mamajek2015] son not: parsek tam olarak (648000/π) au (okundu; türetim: au · 648000 / pi)
- `MEGAPARSEK` = 3.08567758149e+22 m (kesin) [Mamajek2015] son not: parsek tam olarak (648000/π) au (okundu; türetim: 1e6 · parsek)
- `JULYEN_YILI_SANIYE` = 31557600 s (kesin) [ERFA2023] erfam.h ERFA_DJY · ERFA_DAYSEC (okundu; türetim: 365.25 · 86400)

## Kaynak başına fonksiyon sayısı

- [Abramowicz2013]: 11
- [Armitage2011]: 3
- [Astropy2022]: 8
- [Babusiaux2018]: 2
- [Bardeen1972]: 3
- [BertinArnouts1996]: 1
- [Bludman2010]: 13
- [Brout2022]: 8
- [CODATA2022]: 3
- [Carroll2019]: 1
- [Chabrier2003]: 8
- [Cohen2003]: 3
- [CraigChambers2026]: 3
- [EHT2019]: 4
- [ERFA2023]: 27
- [FITS2018]: 3
- [ForemanMackey2013]: 3
- [Hogg1999]: 17
- [HoggBovyLang2010]: 3
- [Howell2006]: 3
- [IRAF2026]: 1
- [Inayoshi2020]: 5
- [Kennicutt1998]: 2
- [Kennicutt2012]: 1
- [Kippenhahn2012]: 1
- [Komossa2015]: 2
- [Kroupa2001]: 8
- [Krumholz2015]: 17
- [Liu1995]: 11
- [Lupton2004]: 3
- [Mamajek2015]: 2
- [McKee2007]: 21
- [MerlineHowell1995]: 1
- [MurrayCorreia2010]: 4
- [Page1976]: 1
- [Page2005]: 5
- [Paxton2013]: 2
- [Prsa2016]: 4
- [SExtractorBelge2026]: 2
- [Salpeter1955]: 4
- [Sathyaprakash2009]: 5
- [Scolnic2022]: 1
- [SkyView2026]: 1
- [TAP2019]: 2
- [Vallenari2023]: 1
- [VanderPlas2018]: 4
- [Vehtari2021]: 2
- [Wright2010]: 3
- [Zinnecker2007]: 1

## Kaynağı okunmayı bekleyenler

- `kozmos.arsiv.gaia.yakin_yildizlar`: [Vallenari2023] Gaia DR3 gaia_source tablosu
- `kozmos.fotometri.kaynak.aciklik_fotometrisi`: [Howell2006] Bölüm 5: açıklık fotometrisi
- `kozmos.fotometri.kaynak.kaynak_bul`: [BertinArnouts1996] saptama: eşik üstü bağlı pikseller
- `kozmos.gozlem.dedektor.sinyal_gurultu_orani`: [MerlineHowell1995] arka plan kestirimi terimi (1 + n_pix/n_B); [Howell2006] Bölüm 4.4: CCD denklemi
- `kozmos.gozlem.dedektor.yildiz_alani`: [Howell2006] Bölüm 5: nokta dağılım fonksiyonu (dairesel Gauss yaklaşımı)
- `kozmos.karadelik.temel.hawking_buharlasma_suresi_s`: [Page1976] Hawking ışınımıyla kütle kaybı ve ömür (greybody düzeltmeli kesin sonuç orada)
- `kozmos.karadelik.temel.isco_isinim_verimliligi`: [Bardeen1972] ISCO'daki özgül enerji Ẽ_ms
- `kozmos.karadelik.temel.isco_ozgul_enerji_kerr`: [Bardeen1972] ISCO'daki özgül enerji Ẽ_ms = (1 − 2/(3 r̃_ms))^{1/2}
- `kozmos.karadelik.temel.isco_yaricapi_kerr_rg`: [Bardeen1972] ISCO formülü r_ms = M{3 + Z2 ∓ [(3 − Z1)(3 + Z1 + 2Z2)]^{1/2}} (retrograde için + işareti)
- `kozmos.karadelik.temel.yercekimsel_kirmiziya_kayma`: [Carroll2019] Schwarzschild metriği, g_tt = −(1 − 2GM/rc²); statik gözlemci
- `kozmos.yildiz.imf.salpeter_dn_dm`: [Salpeter1955] özgün makale (eğim değeri)
- `kozmos.yildiz.imf.salpeter_kutle_orani_ustunde`: [Salpeter1955] özgün makale (eğim değeri)
- `kozmos.yildiz.imf.salpeter_ornekle`: [Salpeter1955] özgün makale (eğim değeri)
- `kozmos.yildiz.imf.salpeter_ortalama_kutle`: [Salpeter1955] özgün makale (eğim değeri)
- `kozmos.yildiz.olcekleme.nukleer_zaman_olcegi_yil`: [Kippenhahn2012] nükleer zaman ölçeği tanımı (bölüm ve denklem numarasına bakılmadı)
