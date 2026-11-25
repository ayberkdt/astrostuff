# Yazılım ekosistemi

Sürüm ve tarih bilgileri 25 Kasım 2026'da PyPI JSON API'sinden `kaynakca/dogrula.py` ile okunmuştur; tam liste [../kaynakca/DOGRULAMA_OZETI.md](../kaynakca/DOGRULAMA_OZETI.md) dosyasındadır. Bakım durumu ve atıf bilgileri altı kollu bir web taramasından gelir; taramanın sayfayı açarak okuduğu bilgi ile yalnızca arama özetinde gördüğü bilgi ayrı işaretlenmiştir. Bu belgede "tarama" sözcüğüyle anılan bilgiler o taramanın bildirimidir ve ayrıca doğrulanmamıştır.

Her paket için projedeki rolü yazılıdır:

- **kâhin:** aynı hesabı yapan bağımsız uygulama; testte karşılaştırılır.
- **kullan:** yeniden yazılmaz, bağımlılık olarak kullanılır.
- **incele:** tasarımı ya da algoritması örnek alınır.

## 1. Çekirdek

- `astropy` 8.0.3 (17 Kasım 2026). Birimler, sabitler, koordinatlar, zaman, WCS, FITS, tablo, kozmoloji, modelleme, istatistik ve zaman serisi alt paketleri tek dağıtımdadır. Atıf: üç proje makalesi [Astropy2013; Astropy2018; Astropy2022]. Rol: hemen her modülün kâhini.
- `pyerfa` 2.0.1.5 (11 Kasım 2024). IAU SOFA algoritmalarının ERFA uyarlamasının Python sarmalayıcısı. SOFA'nın güncel sürümü 11 Ekim 2023 tarihlidir (iausofa.org, taramada okundu) [SOFA2023]. Rol: zaman ve koordinat için birincil kâhin; SOFA yemek kitapları algoritma kaynağı.
- `numpy` 2.5.5, `scipy` 1.18.1. Rol: çalışma zamanı bağımlılığı [Harris2020; Virtanen2020].

## 2. Arşiv erişimi

- `astroquery` 0.4.11 (20 Eylül 2025); arşiv ve hizmetlere sorgu [Ginsburg2019]. Kararlı sürüm bir yılı aşkın süredir yenilenmemiştir; taramaya göre geliştirme ön sürümleri sürmektedir.
- `pyvo` 1.10.0 (4 Kasım 2026); TAP, SIA, SSA istemcisi.
- `astrocut` 1.5.1, `mocpy` 0.21.0, `astropy-healpix` 2.0.1, `healpy` 1.20.1, `lsdb` 0.11.2.

Rol: kâhin (`arsiv/tap.py` için) ve yedek erişim yolu.

## 3. Konum astronomisi, efemeris, yörünge

- `skyfield` 1.55 (7 Ağustos 2026), `jplephem` 2.24: JPL DE efemerisleriyle konum hesabı. Rol: efemeris kâhini.
- `spiceypy` 8.4.0 (11 Kasım 2026): NAIF SPICE araç takımının sarmalayıcısı. Rol: Düzey 3 kâhini.
- `sgp4` 2.27: uydu ilerletici. Rol: kullan.
- `ephem` 4.2.1: eski kuşak kitaplık; hâlâ sürüm çıkıyor.
- `rebound` 5.2.3 (18 Kasım 2026), `reboundx` 5.1.0: N cisim integratörleri [ReinLiu2012; ReinTamayo2015; ReinSpiegel2015]. Rol: kâhin ve incele.
- `poliastro` 0.17.0 (10 Temmuz 2022): taramaya göre GitHub deposu 14 Ekim 2023'te arşivlenmiştir. Çatalı `hapsira` 0.18.0 (24 Aralık 2023) o tarihten beri PyPI'da yeni sürüm çıkarmamıştır. İkisi de bakımlı bir kâhin sayılmaz; yörünge mekaniği testleri REBOUND, Skyfield ve kaynak örneklerine dayanır.
- `galpy` 1.12.1 (2 Kasım 2026), `gala` 1.12.0: gökada dinamiği. Rol: `gokada` kâhini.
- JPL Horizons: bağımsız efemeris hizmeti. DE440/DE441 güncel gezegen efemerisidir [Park2021]; tarama JPL sayfasında daha yeni bir sürüm adı görmemiştir.

## 4. Görüntü indirgeme, fotometri, astrometri

- `ccdproc` 2.5.1 (5 Temmuz 2025): CCD indirgeme. Taramaya göre GitHub sürüm başlığı bunu "son 2.x sürümü" olarak adlandırır; 3.x'in durumu doğrulanmamıştır.
- `astroscrappy` 1.3.0: L.A.Cosmic uygulaması [vanDokkum2001].
- `photutils` 3.1.1 (16 Ekim 2026): kaynak saptama, açıklık ve PSF fotometrisi. Ana sürüm 3'e geçmiştir; 2.x'e göre arayüz değişiklikleri incelenmemiştir.
- `sep` 1.4.1 (18 Şubat 2025): SExtractor çekirdeğinin kitaplık biçimi [Barbary2016; BertinArnouts1996]. Taramaya göre proje el değiştirmiştir (özgün depo 1.2.1'e kadar, güncel geliştirme `sep-developers` altında).
- `astroalign` 2.6.2, `reproject` 0.21.0, `regions` 0.13.
- `twirl` 0.6.0 (26 Ekim 2026): Gaia ile saf Python astrometrik çözüm. `astrometry.net`: kör çözücü [Lang2010].
- `stdpipe` 0.5.0: geçici kaynak hattı.
- SExtractor, SCAMP, SWarp: taramaya göre resmî sayfalarda son sürümler 2006 ve 2010 tarihlidir; ardılı SourceXtractor++'tır. Python işinde `sep` ve `photutils` tercih edilir.
- IRAF: taramaya göre topluluk dağıtımı 2.18.2'dir ve topluluk sayfası yeni projeler için Astropy'yi önerir.

Rol: `ccdproc`, `photutils`, `sep`, `astroscrappy` kâhin; `astrometry.net` ve `twirl` ilk aşamada kullan.

## 5. Tayf

- `specutils` 2.4.0, `specreduce` 1.9.0: tayf kapları, çıkarım ve dalga boyu kalibrasyonu.
- `pypeit` 3.0.0 (9 Kasım 2026): yarı otomatik tayf indirgeme hattı.
- `synphot` 1.9.0, `speclite` 1.0.0: sentetik fotometri ve filtreler.
- `dust-extinction` 1.7, `extinction` 0.4.10, `dustmaps` 1.0.14: sönümleme yasaları ve toz haritaları.
- `ppxf` 9.5.3 (24 Kasım 2026): tam tayf uydurma.

Rol: kâhin.

## 6. Zaman serisi ve ötegezegen

- `lightkurve` 2.6.0: Kepler ve TESS verisi.
- `transitleastsquares` 2.3.0 (12 Kasım 2026), `wotan` 2.1.1 (18 Kasım 2026): geçiş arama ve eğilim giderme [HippkeHeller2019].
- `batman-package` 2.5.3: geçiş modeli [Kreidberg2015; MandelAgol2002].
- `exoplanet` 0.6.0 (Nisan 2024), `starry` 1.2.0 (Ekim 2021): uzun süredir sürüm çıkmamıştır.
- `radvel` 1.6.7, `stingray` 2.3.2, `celerite2` 0.3.3, `george` 0.4.4, `tinygp` 0.3.1.

Rol: `astropy.timeseries`, `transitleastsquares`, `batman` kâhin.

## 7. Kozmoloji

- `astropy.cosmology`: uzaklık hesapları. Rol: birincil kâhin.
- `camb` 2.0.5, `classy` 3.4.3.0: Boltzmann kodları. Rol: kullan (güç tayfı ve ses ufku girdisi).
- `cobaya` 3.6.2, `getdist` 1.7.7: örnekleme çatısı ve zincir çözümlemesi.
- `sncosmo` 2.13.2 (23 Kasım 2026): süpernova ışık eğrisi uydurma [Guy2007; Kenworthy2021]. Rol: kullan.
- `pyccl` 3.3.6, `colossus` 1.4.0: yapı oluşumu ve hale hesapları.

## 8. İstatistik ve çıkarım

- `emcee` 3.1.6 (19 Nisan 2024) [ForemanMackey2013], `zeus-mcmc` 2.5.4 (Ocak 2023).
- `dynesty` 3.2.0 (10 Kasım 2026) [Speagle2020], `nautilus-sampler` 1.0.6, `ultranest` 4.6.3.
- `pymc` 6.4.0, `numpyro` 0.22.0, `blackjax` 1.7.1, `cmdstanpy` 1.3.2: gradyan tabanlı örnekleme.
- `arviz` 1.3.0, `corner` 2.3.0: tanı ve görselleştirme.
- `astroML` 1.0.2.post1 (Mart 2022): Ivezić ve arkadaşlarının kitabının eşlik eden kodu [Ivezic2020]; uzun süredir sürüm çıkmamıştır, yöntem kaynağı olarak değerlidir.
- `scikit-learn` 1.9.1, `iminuit` 2.33.0, `lmfit` 1.3.4, `uncertainties` 3.2.3, `sbi` 0.27.0, `jax` 0.12.1.

Rol: `emcee` ve `dynesty` kâhin; `scikit-learn` kullan; NumPyro ya da PyMC ileri aşamada kullan.

## 9. Astrofizik alan kodları

- Yıldız: MESA (taramaya göre belgelerindeki güncel sürüm 26.4.1) [Paxton2011], MIST ve PARSEC ızgaraları [Choi2016; Bressan2012], `fsps` 0.5.0, `astro-prospector` 1.4.1, `bagpipes` 1.4.1.
- Benzetim çözümlemesi: `yt` 4.4.4, `pynbody` 2.8.0.
- Yüksek enerji: `gammapy` 2.1, `sherpa` 4.19.0, `naima` 0.11.0.
- Çekimsel dalga: `gwpy` 4.1.0, `pycbc` 2.11.0, `bilby` 2.8.2 [Abbott2016].
- Pulsar zamanlama: `pint-pulsar` 1.1.8.
- Güneş ve plazma: `sunpy` 8.0.1, `plasmapy` 2026.2.0.
- Görelilik: `einsteinpy` 0.5.0.
- Tayf çizgisi ve ışınım: `pyneb` 1.1.32, `radis` 0.17.1.
- `tardis-sn`: PyPI'daki 1.0.1 sürümü 2015 tarihlidir; proje PyPI üzerinden dağıtılmıyor olabilir, güncel kurulum yolu incelenmemiştir.

Rol: çoğu ileri aşama için incele; yeniden yazılmaz.

## 10. Tarama kapsamı dışında kalanlar

Tarama aşağıdakileri okuyamadı ya da ele almadı; envanterin bu kısmı eksiktir: radyo astronomi (CASA, `spectral-cube`, `pyuvdata`), ışınım aktarımı kodları (RADMC-3D, Cloudy, SKIRT), atom ve molekül veritabanları (CHIANTI, HITRAN, ExoMol), hidrodinamik kodları (GADGET-4, AREPO, Athena++), ötegezegen atmosfer kodları (petitRADTRANS, PICASO), tayfsal enerji dağılımı uydurma (CIGALE), LSST Science Pipelines, DRAGONS. Bu alanlar yol haritasının ileri aşamalarına girdiğinde ayrıca taranmalıdır.

## 11. Sonuç

- Python astronomi ekosistemi olgundur: hemen her temel hesap için bakımlı bir kâhin vardır. Projenin katkısı yeni işlev değil, her denklemi kaynağına bağlayan, okunabilir ve doğrulanmış bir uygulamadır.
- Bakımı durmuş ya da belirsiz paketler: `poliastro`, `hapsira`, `astroML`, `starry`, `exoplanet`, `zeus-mcmc`, `astroplan`. Bunlara dayanan testler kaynak örnekleriyle desteklenmelidir.
- 2026'da ana sürüm değiştiren paketler (`astropy` 8, `photutils` 3, `pypeit` 3, `pymc` 6, `sunpy` 8): kâhin testleri belirli sürüme sabitlenir ve sürüm referans çıktının başlığına yazılır.
- SOFA/ERFA ikilisi konum astronomisinin fiilî standardıdır; algoritma kaynağı ve kâhin olarak ilk başvurulacak yerdir.
