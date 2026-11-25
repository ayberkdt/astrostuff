# Veri erişimi ve biçimler

Hedef modüller: `io/`, `arsiv/`.

Arşivlerin güncel sürüm durumu [../02_saha_arastirmasi/03_veri_arsivleri.md](../02_saha_arastirmasi/03_veri_arsivleri.md) dosyasındadır.

## 1. FITS

Astronomi verisinin ortak dosya biçimidir. Güncel standart sürüm 4.0'dır; dergide yayımlanmış tanım sürüm 3.0'ı anlatır [Pence2010]. Sürüm 4.0'ın dergi künyesi yoktur; NASA FITS destek bürosunun belgesi anılır [FITS2018].

- Yapı: 2880 baytlık bloklar; başlık (80 karakterlik kartlar: anahtar, değer, yorum) ve veri birimi; birincil HDU ve uzantılar (görüntü, ikili tablo, ASCII tablo).
- Zorunlu anahtarlar: `SIMPLE`, `BITPIX`, `NAXIS`, `NAXISn`, `END`. Ölçekleme: `BSCALE`, `BZERO` (işaretsiz 16 bit tamsayının saklanma yolu).
- Kurallar: `CONTINUE` uzun dizgiler, `HIERARCH`, çok uzantılı dosyalar, sıkıştırılmış görüntü (tile compression), `CHECKSUM`/`DATASUM`.
- Zaman anahtarları: `DATE-OBS`, `MJD-OBS`, `TIMESYS`, `EXPTIME`; poz başlangıcı ile orta noktasının ayrımı [Rots2015].
- Dünya koordinat sistemi: [../03_matematik/01_kuresel_astronomi_ve_donusumler.md](../03_matematik/01_kuresel_astronomi_ve_donusumler.md), Bölüm 8 [GreisenCalabretta2002; CalabrettaGreisen2002; Greisen2006].

**Uygulama kapsamı.** Okuyucu ve yazıcı sıfırdan yazılır: başlık ayrıştırma, birincil görüntü, görüntü uzantısı, ikili tablo. Sıkıştırılmış görüntü ve değişken uzunluklu dizi sütunları ikinci aşamadır. Kâhin: `astropy.io.fits` ve `fitsio`. Test: aynı dosyanın iki okuyucuyla bayt düzeyinde aynı diziyi vermesi; yazılan dosyanın `astropy` doğrulayıcısından geçmesi.

Gerçek başlıklar standarda uymaz: amatör yazılımlar ve eski aygıtlar eksik ya da hatalı anahtar yazar. Okuyucu hoşgörülü, yazıcı katı olur; her sapma uyarı olarak kaydedilir.

## 2. Tablo ve katalog biçimleri

- VOTable (XML tabanlı; güncel öneri 1.5), CSV/ECSV (üstveri taşıyan CSV), FITS ikili tablo, Parquet (büyük kataloglar), HDF5 (benzetim çıktıları).
- Sütun üstverisi: birim, UCD (içerik tanımlayıcı), açıklama; birimsiz sütun okunurken uyarı üretir.
- Büyük kataloglar: HEALPix bölümlü Parquet (HATS biçimi) ve tembel değerlendirme.

## 3. Sanal Gözlemevi protokolleri

IVOA standartları arşivlere ortak erişim sağlar. Araştırmada IVOA belge dizininden okunan öneri sürümleri: TAP 1.1 (2019), ADQL 2.1 (2023), SIA 2.0 (2015), SSA 1.1 (2012), Simple Cone Search 1.03 (2008), VOTable 1.5 (2025), HiPS 1.0 (2017), MOC 2.0 (2022), DataLink 1.1 (2023).

- **TAP:** HTTP üzerinden katalog sorgulama; eşzamanlı ve eşzamansız iş modeli (iş oluştur, durumu yokla, sonucu al).
- **ADQL:** SQL'in astronomi uzantısı; geometrik işlevler (`CONTAINS`, `POINT`, `CIRCLE`, `DISTANCE`), çapraz eşleme.
- **Cone Search, SIA, SSA:** konuma göre katalog, görüntü ve tayf arama.
- **DataLink:** bir kayda bağlı veri ürünlerinin bulunması (Gaia tayfları ve ışık eğrileri bu yolla alınır).
- **HiPS ve MOC:** hiyerarşik görüntü döşemeleri ve gökyüzü kapsama haritaları.

**Uygulama kapsamı.** `arsiv/tap.py` TAP eşzamanlı ve eşzamansız istemcisini standart kütüphane üzerinde yazar; VOTable ayrıştırıcı `io/votable.py` içindedir. Kâhin: `pyvo` ve `astroquery` [Ginsburg2019].

## 4. Arşive özgü erişim

Her arşiv için ince bir modül: uç nokta, tablo adları, kalite bayrakları, önerilen kesimler ve atıf metni.

- **Gaia:** TAP; `gaia_source` sütunları, `ruwe` ve diğer kalite ölçütleri, paralaks sıfır noktası [Vallenari2023; Lindegren2021].
- **Pantheon+SH0ES:** GitHub veri sürümünden uzaklık modülü tablosu ve kovaryans matrisleri [Scolnic2022; Brout2022].
- **MAST:** TESS ve Kepler ışık eğrileri, hedef piksel dosyaları, tam kare kesitleri [Ricker2015].
- **SDSS, DESI:** tayflar ve kırmızıya kaymalar [York2000].
- **ZTF:** ışık eğrileri [Bellm2019].
- **NASA Exoplanet Archive:** TAP ile gezegen parametreleri. Arşivin belgelerine göre eski `nstedAPI` arayüzündeki tabloların çoğu kaldırılmıştır; yeni kod TAP'ı hedefler.
- **JPL Horizons ve SBDB:** efemeris ve küçük cisim elemanları.
- **SIMBAD, VizieR, NED:** ad çözümleme, yayımlanmış tablolar, gökada dışı veritabanı.
- **AAVSO, Minor Planet Center:** değişen yıldız gözlemleri ve astrometri gönderimi.

## 5. Önbellek ve kaynak kaydı

- Her indirme yerel önbelleğe yazılır; anahtar sorgu metninin özeti, değer veri ile birlikte tutulan üstveridir: URL, sorgu, tarih, dosya özeti (SHA-256), arşivin veri sürümü.
- Aynı sorgu ikinci kez ağa çıkmaz; yenileme açık bir argümandır.
- Arşiv veri sürümü (Gaia DR3, SDSS DR-n) sorguya gömülür; "en son" takma adı kullanılmaz, çünkü sonuç zamanla değişir.
- Ağ katmanı hız sınırına uyar, yeniden dener ve hata durumunda kısmi dosya bırakmaz.

## 6. Kendi gözlem verisi

Gözlem gecesi dizin düzeni ve dosya adlandırma kuralı; başlık anahtarlarının bir aygıt tanım dosyasıyla ortak şemaya eşlenmesi (kazanç, okuma gürültüsü, piksel ölçeği, filtre adları); gece günlüğü. Aygıt tanımı bir kez yazılır, indirgeme hattı her gece ondan okur.
