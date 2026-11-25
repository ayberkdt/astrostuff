# Uygulama durumu

Bu dosya `python kaynakca/kutuk_raporu.py` ile kaynak kütüğünden üretilir; elle düzenlenmez.

- Kaynağa bağlı açık fonksiyon: 62
- Bütün kayıtları kaynak metninden okunmuş fonksiyon: 50
- Kaynak kaydı: 64 (okundu: 52, bekliyor: 12)
- Kaynaklı sabit: 35 (teyit bekleyen: 2)
- Test fonksiyonu: 80 (parametreli ve özellik testleri tek sayılır)
- Kaynak kuralından muaf altyapı fonksiyonu: 1

`okundu`: denklem kod yazılırken kaynağın metninden okundu. `bekliyor`: denklem hatırlanan biçimiyle
yazıldı, testlerle sınandı, ama kaynağın metni henüz açılmadı.

## Fonksiyonlar

### `kozmos.arsiv.pantheon`

- `indir`: [Scolnic2022] veri deposu, 4_DISTANCES_AND_COVAR (okundu)
- `kovaryans_oku`: [Brout2022] veri deposu README: .cov biçimi (okundu)
- `tablo_oku`: [Brout2022] veri deposu README: sütun tanımları (okundu)
- `yukle`: [Brout2022] veri deposu README (okundu)

### `kozmos.efemeris.kepler`

- `dis_merkezli_anomali`: [MurrayDermott1999] Bölüm 2.4, Kepler denklemi (bekliyor)
- `gercek_anomali`: [MurrayDermott1999] Bölüm 2.4, gerçek anomali (bekliyor)
- `yorunge_donemi_s`: [MurrayDermott1999] Bölüm 2.3, Kepler'in üçüncü yasası (bekliyor)
- `yorunge_hizi_m_s`: [MurrayDermott1999] Bölüm 2.4, vis-viva (bekliyor)

### `kozmos.isinim.kadir`

- `bolometrik_kadirden_isinim_gucu_w`: [Mamajek2015] M_Bol tanımı; L_0 = 3,0128e28 W (okundu)
- `kara_cisim_akisi`: [CODATA2022] Stefan-Boltzmann constant (okundu)
- `mutlak_bolometrik_kadir`: [Mamajek2015] M_Bol tanımı; L_0 = 3,0128e28 W (okundu)
- `planck_dalgaboyu`: [RybickiLightman1979] Bölüm 1.5, Planck tayfı (bekliyor)
- `planck_frekans`: [RybickiLightman1979] Bölüm 1.5, Planck tayfı (bekliyor)
- `uzaklik_modulu_pc`: [Hogg1999] Denk. 25 (okundu)
- `uzaklik_modulunden_pc`: [Hogg1999] Denk. 25 (okundu)

### `kozmos.istatistik.mcmc`

- `bolunmus_rhat`: [GelmanRubin1992] potansiyel ölçek indirgeme çarpanı (bekliyor)
- `gerdirme_carpani`: [ForemanMackey2013] Denk. goodmanprop (g(z) ∝ 1/√z, z ∈ [1/a, a]) (okundu)
- `oz_ilinti_suresi`: [ForemanMackey2013] Bölüm 3 (öz ilinti süresi) (bekliyor)
- `topluluk_ornekle`: [GoodmanWeare2010] gerdirme hamlesi (bekliyor); [ForemanMackey2013] Denk. proposal, acceptance; Algoritma 3 (paralel gerdirme hamlesi) (okundu)

### `kozmos.istatistik.uyum`

- `dogru_uydur`: [HoggBovyLang2010] TeX etiketi eq:fofx, eq:lsf (okundu)
- `dogrusal_en_kucuk_kareler`: [HoggBovyLang2010] TeX etiketi eq:lsf ve parametre kovaryansı denklemi (okundu)
- `ki_kare`: [HoggBovyLang2010] TeX etiketi eq:chisquared (okundu)

### `kozmos.koordinat.donusum`

- `galaktik_den_icrs`: [ERFA2024] icrs2g.c (okundu)
- `hava_kutlesi_duzlem_paralel`: [ERFA2024] hd2ae.c (z bileşeni) (okundu)
- `icrs_den_galaktik`: [ERFA2024] icrs2g.c (okundu)
- `saat_acisi_den_ufuk`: [ERFA2024] hd2ae.c (okundu)
- `ufuk_den_saat_acisi`: [ERFA2024] ae2hd.c (okundu)

### `kozmos.kozmoloji.hubble_diyagrami`

- `artik_vektoru`: [Brout2022] eq:dmu, eq:dmuprime, eq:dl (okundu)
- `en_iyi_mutlak_kadir`: [Brout2022] eq:likelihood (okundu)
- `log_olabilirlik`: [Brout2022] eq:likelihood, eq:sh0eslikelihood (okundu)
- `veri_hazirla`: [Brout2022] Bölüm 4: z < ~0,01 Hubble diyagramı kullanılmaz; eq:dmuprime (okundu)

### `kozmos.kozmoloji.uzaklik`

- `acisal_cap_uzakligi_mpc`: [Hogg1999] Denk. 18 (okundu)
- `e_fonksiyonu`: [Brout2022] H(z) denklemi (TeX etiketi eq:hz) (okundu); [Hogg1999] Denk. 14 (okundu)
- `enine_esyuruyen_uzaklik_mpc`: [Hogg1999] Denk. 16 (okundu)
- `esyuruyen_hacim_elemani_mpc3`: [Hogg1999] Denk. 28 (okundu)
- `esyuruyen_uzaklik_mpc`: [Hogg1999] Denk. 15 (okundu)
- `geriye_bakis_zamani_gyr`: [Hogg1999] Denk. 30 (okundu)
- `hubble_uzakligi_mpc`: [Hogg1999] Denk. 4 (okundu)
- `hubble_zamani_gyr`: [Hogg1999] Denk. 3 (okundu)
- `isinim_gucu_uzakligi_mpc`: [Hogg1999] Denk. 21 (okundu)
- `uzaklik_modulu`: [Hogg1999] Denk. 25 (okundu)

### `kozmos.temel.aci`

- `acisal_ayriklik`: [ERFA2024] sepp.c (okundu)
- `birim_vektor_den_kuresel`: [ERFA2024] c2s.c (okundu)
- `derece_den_radyan`: [ERFA2024] erfam.h ERFA_DD2R (okundu)
- `eksi_pi_arti_pi`: [ERFA2024] anpm.c (okundu)
- `kuresel_den_birim_vektor`: [ERFA2024] s2c.c (okundu)
- `radyan_dan_derece`: [ERFA2024] erfam.h ERFA_DD2R (okundu)
- `sifir_iki_pi`: [ERFA2024] anp.c (okundu)
- `yaysaniye_den_radyan`: [ERFA2024] erfam.h ERFA_DAS2R (okundu)

### `kozmos.zaman.julyen`

- `dunya_donme_acisi`: [ERFA2024] era00.c (okundu)
- `greenwich_ortalama_yildiz_zamani`: [ERFA2024] gmst06.c (okundu)
- `jd_den_takvim`: [ERFA2024] jd2cal.c (okundu)
- `ortalama_egiklik`: [ERFA2024] obl06.c (okundu)
- `tai_den_tt`: [ERFA2024] erfam.h ERFA_TTMTAI, ERFA_DAYSEC (okundu)
- `takvim_den_jd`: [ERFA2024] cal2jd.c; erfam.h ERFA_DJM0 (okundu)
- `takvim_den_mjd`: [ERFA2024] cal2jd.c (okundu)
- `tt_den_tai`: [ERFA2024] erfam.h ERFA_TTMTAI, ERFA_DAYSEC (okundu)
- `yerel_ortalama_yildiz_zamani`: [ERFA2024] gmst06.c; boylam doğuya artı (bekliyor)

### `kozmos.zamanserisi.periyodogram`

- `frekans_izgarasi`: [VanderPlas2018] Bölüm 7.1 (frekans ızgarası), eq:n-eval (bekliyor)
- `lomb_scargle`: [VanderPlas2018] eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized (okundu)
- `lomb_scargle_klasik`: [VanderPlas2018] eq:lomb-scargle-periodogram, eq:tau-def (okundu)
- `yanlis_alarm_olasiligi`: [VanderPlas2018] eq:LS-prob-z ve eq:FAP-neff çevresi (bekliyor)

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
- `ASTRONOMIK_BIRIM` = 149597870700 m (kesin) [ERFA2024] erfam.h ERFA_DAU (okundu)
- `GUN_SANIYE` = 86400 s (kesin) [ERFA2024] erfam.h ERFA_DAYSEC (okundu)
- `JULYEN_YILI_GUN` = 365.25 gün (kesin) [ERFA2024] erfam.h ERFA_DJY (okundu)
- `JULYEN_YUZYILI_GUN` = 36525 gün (kesin) [ERFA2024] erfam.h ERFA_DJC (okundu)
- `J2000_JD` = 2451545 gün (kesin) [ERFA2024] erfam.h ERFA_DJ00 (okundu)
- `MJD_SIFIR_JD` = 2400000.5 gün (kesin) [ERFA2024] erfam.h ERFA_DJM0 (okundu)
- `TT_EKSI_TAI` = 32.184 s (kesin) [ERFA2024] erfam.h ERFA_TTMTAI (okundu)
- `PARSEK` = 3.08567758149e+16 m (kesin) [Mamajek2015] parsek tanımı notu (bekliyor; türetim: au · 648000 / pi)
- `MEGAPARSEK` = 3.08567758149e+22 m (kesin) [Mamajek2015] parsek tanımı notu (bekliyor; türetim: 1e6 · parsek)
- `JULYEN_YILI_SANIYE` = 31557600 s (kesin) [ERFA2024] erfam.h ERFA_DJY · ERFA_DAYSEC (okundu; türetim: 365.25 · 86400)

## Kaynak başına fonksiyon sayısı

- [Brout2022]: 8
- [CODATA2022]: 1
- [ERFA2024]: 22
- [ForemanMackey2013]: 3
- [GelmanRubin1992]: 1
- [GoodmanWeare2010]: 1
- [Hogg1999]: 12
- [HoggBovyLang2010]: 3
- [Mamajek2015]: 2
- [MurrayDermott1999]: 4
- [RybickiLightman1979]: 2
- [Scolnic2022]: 1
- [VanderPlas2018]: 4

## Kaynağı okunmayı bekleyenler

- `kozmos.efemeris.kepler.dis_merkezli_anomali`: [MurrayDermott1999] Bölüm 2.4, Kepler denklemi
- `kozmos.efemeris.kepler.gercek_anomali`: [MurrayDermott1999] Bölüm 2.4, gerçek anomali
- `kozmos.efemeris.kepler.yorunge_donemi_s`: [MurrayDermott1999] Bölüm 2.3, Kepler'in üçüncü yasası
- `kozmos.efemeris.kepler.yorunge_hizi_m_s`: [MurrayDermott1999] Bölüm 2.4, vis-viva
- `kozmos.isinim.kadir.planck_dalgaboyu`: [RybickiLightman1979] Bölüm 1.5, Planck tayfı
- `kozmos.isinim.kadir.planck_frekans`: [RybickiLightman1979] Bölüm 1.5, Planck tayfı
- `kozmos.istatistik.mcmc.bolunmus_rhat`: [GelmanRubin1992] potansiyel ölçek indirgeme çarpanı
- `kozmos.istatistik.mcmc.oz_ilinti_suresi`: [ForemanMackey2013] Bölüm 3 (öz ilinti süresi)
- `kozmos.istatistik.mcmc.topluluk_ornekle`: [GoodmanWeare2010] gerdirme hamlesi
- `kozmos.zaman.julyen.yerel_ortalama_yildiz_zamani`: [ERFA2024] gmst06.c; boylam doğuya artı
- `kozmos.zamanserisi.periyodogram.frekans_izgarasi`: [VanderPlas2018] Bölüm 7.1 (frekans ızgarası), eq:n-eval
- `kozmos.zamanserisi.periyodogram.yanlis_alarm_olasiligi`: [VanderPlas2018] eq:LS-prob-z ve eq:FAP-neff çevresi
