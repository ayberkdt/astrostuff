# Hubble sabiti ve kozmoloji: veri ve güncel durum

Bu belgedeki her bilginin yanında nasıl elde edildiği yazılıdır: **okundu** (25 Kasım 2026'da kaynağından doğrudan okundu), **tarama-sayfa** (web taraması sayfayı açarak bildirdi), **tarama-özet** (tarama yalnızca arama sonucu özetinde gördü). Son ikisi ayrıca doğrulanmamıştır.

## 1. Pantheon+SH0ES veri sürümü

Projenin birincil veri setidir.

- Konum: `https://github.com/PantheonPlusSH0ES/DataRelease`, `Pantheon+_Data/4_DISTANCES_AND_COVAR/` (okundu; GitHub API).
- Dosyalar (okundu):
  - `Pantheon+SH0ES.dat` (579 kB): 1701 satır, 47 sütun.
  - `Pantheon+SH0ES_STAT+SYS.cov` (33 MB): ilk satırı boyut (`1701`), ardından satır başına bir matris elemanı.
  - `Pantheon+SH0ES_STATONLY.cov` (32 MB).
  - `README`.
- Tablonun içeriği (okundu): 1701 ışık eğrisi satırı, 1543 tekil süpernova kimliği (`CID`); `zHD` aralığı 0,00122-2,26137; `IS_CALIBRATOR = 1` olan 77 satır.
- Analizde kullanılacak sütunlar: `zHD` (Hubble diyagramı kırmızıya kayması), `zHEL`, `m_b_corr` ve `m_b_corr_err_DIAG` (standartlaştırılmış kadir), `MU_SH0ES`, `CEPH_DIST` (Sefeid konak uzaklık modülü), `IS_CALIBRATOR`, `USED_IN_SH0ES_HF`, `x1`, `c`, `HOST_LOGMASS`, `VPEC`, `MWEBV`. Sütunların tanımı depodaki `README` dosyasından ve makalelerden okunacaktır; bu belgedeki açıklamalar sütun adlarından çıkarılmıştır.
- Lisans (okundu): depoda lisans dosyası tanımlı değildir (GitHub API `license: null`). Veri yeniden dağıtılmaz; betik indirir. Atıf: [Scolnic2022; Brout2022] ve merdiven için [Riess2022].
- Depoya son gönderim 21 Aralık 2022 tarihlidir (okundu); veri durağandır, yine de indirilen dosyaların özeti kaydedilir.

## 2. Diğer süpernova derlemeleri

- Pantheon (2018): arXiv:1710.00845 (tarama-sayfa).
- Union3: arXiv:2311.12098; 2087 süpernova (tarama-özet). Union3.1: arXiv:2601.19424 (tarama-özet). Veri konumu doğrulanmadı.
- DES-SN5YR: arXiv:2401.02929 (tarama-sayfa); yeniden kalibrasyon arXiv:2511.07517 (tarama-özet). Veri konumu doğrulanmadı.
- Open Supernova Catalog: taramaya göre proje sona ermiş, verisi arşivlenmiştir (tarama-özet).

Derlemeler arası fark karanlık enerji sonuçlarını etkilemektedir; ikinci bir derlemeyle çapraz denetim yol haritasının ileri adımıdır.

## 3. Uzaklık merdiveni verisi

- SH0ES Sefeid fotometrisi: tabloların konumu doğrulanmadı [Riess2022]. Pantheon+SH0ES tablosundaki `CEPH_DIST` sütunu konak uzaklıklarını hazır verir; Sefeid düzeyinde yeniden analiz ayrı bir projedir.
- CCHP TRGB ve JAGB uzaklıkları: veri konumu doğrulanmadı [Freedman2019; Freedman2025].
- Cosmicflows-4: arXiv:2209.11238 (tarama-sayfa); veri konumu doğrulanmadı.
- NED-D: doğrulanmadı.
- Hubble'ın 1929 tablosu: özgün makaledeki gökada uzaklık ve hız tablosu [Hubble1929]. Taramada makine okunur bir kopya bulunamadı; tablo makaleden elle aktarılacak ve aktarım ikinci bir okumayla denetlenecektir.

## 4. BAO, CMB ve bağımsız ölçümler

- DESI DR1 BAO kozmolojisi [DESI2025] (künye Crossref ile doğrulandı). DESI DR2 BAO: arXiv:2503.14738 [DESI2025DR2] (künye arXiv API ile doğrulandı).
- DESI açık veri: taramaya göre DR1 19 Mart 2025'te yayımlanmıştır, erişim toplu dosya indirmedir (tarama-sayfa).
- Planck 2018 kozmolojik parametreleri [Planck2020]; zincirler Planck Legacy Archive'dadır, tarama arşiv sayfasını okuyamadı.
- ACT DR6, SPT-3G, kozmik kronometre derlemeleri, mercek zaman gecikmeleri: taranmadı.
- Standart siren: GW170817 [Abbott2017].

## 5. Hubble sabitinin güncel değerleri

- SH0ES (Sefeid + SN Ia): `73,04 ± 1,04 km s⁻¹ Mpc⁻¹` [Riess2022] (tarama-sayfa: arXiv:2112.04510).
- Pantheon+ ile SH0ES birlikte: `73,5 ± 1,1` (`w0CDM`), `73,3 ± 1,1` (`w0waCDM`) [Brout2022] (tarama-sayfa).
- CCHP, yalnızca TRGB: `70,39 ± 1,22 ± 1,33 ± 0,70` [Freedman2025] (tarama-sayfa). Yalnızca JWST verisiyle TRGB `68,81` ve JAGB `67,80` (tarama-özet). Makalenin sürümleri arasında değerler değişmiştir.
- SH0ES HST+JWST birleşimi `73,18 ± 0,88` (tarama-özet; birincil makale bulunamadı).
- Erken evren (Planck ΛCDM) değeri bu çalışmada kaynağından okunmadı.

Taramanın vardığı sonuç: gerilim en yeni yayınlarda da çözülmemiştir; Sefeid ve TRGB merdivenleri arasındaki fark ayrı bir açık tartışmadır; DESI DR2, BAO ve CMB birleşiminin dinamik karanlık enerjiyi 3,1σ düzeyinde yeğlediğini bildirmektedir (tarama-özet).

Derleme kaynakları: [DiValentino2021] (künye doğrulandı); CosmoVerse raporu arXiv:2504.01669 (tarama-özet).

## 6. Proje için sonuçlar

1. Pantheon+SH0ES tek başına uçtan uca bir `H0` analizi için yeterlidir: Hubble akışı süpernovaları, Sefeid çapalı kalibratörler ve tam kovaryans aynı sürümdedir.
2. Hedef, yayımlanan sonucu yeniden üretmektir. Yeniden üretim tutmadan başka veri eklenmez.
3. Güncel değerler hızla değişmektedir. Projede herhangi bir `H0` değeri "güncel" diye kodlanmaz; her değer künyesi ve sürümüyle anılır.
4. Yapılacak okuma: [Riess2022] ve [Brout2022] makalelerinin tam metni, veri deposunun `README` dosyası, 2025-2026 derlemeleri.
