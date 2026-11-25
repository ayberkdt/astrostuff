# Veri arşivleri

Arşivlerin sürüm durumu web taramasından gelir ve 25 Kasım 2026 tarihlidir. **tarama-sayfa** taramanın sayfayı açarak bildirdiği, **doğrulanmadı** taramanın okuyamadığı ya da ele almadığı bilgidir. Bu belgedeki hiçbir sürüm bilgisi ayrıca doğrulanmamıştır; bir arşive ilk kez bağlanılırken sürüm arşivin kendi sayfasından okunur.

## Astrometri ve yıldız katalogları

- **Gaia** (ESA). Güncel sürüm DR3'tür [Vallenari2023]. ESA'nın DR4 sayfası 14 Eylül 2026 güncellemesinde verinin "yayımlanacağını" gelecek zamanla yazmaktadır ve tarih vermemektedir (tarama-sayfa). Erişim: TAP/ADQL ve DataLink, `gea.esac.esa.int/archive`. Projede kullanım: astrometrik ve fotometrik referans katalog, HR diyagramı, küme üyeliği, Sefeid paralaksları. DR4 çıktığında sorgulara gömülü sürüm adı değişir; testler DR3'e sabitlenir.
- **2MASS, WISE, Pan-STARRS:** doğrulanmadı [Skrutskie2006].

## Tayf taramaları

- **SDSS.** Tarama `sdss.org/dr20` sayfasında "Data Release 20" adını görmüştür (tarama-sayfa); yayım tarihi ve erişim ayrıntıları okunmamıştır [York2000].
- **DESI.** DR1 19 Mart 2025, erken veri sürümü 13 Haziran 2023; erişim toplu dosya indirme (tarama-sayfa).

## Görüntüleme taramaları ve zaman alanı

- **Vera C. Rubin Gözlemevi (LSST).** Data Preview 1: devreye alma verisi, yaklaşık 7 alan, 2024 sonu gözlemleri. "Early Data Preview 2": Nisan 2025-Ocak 2026 gözlemleri (tarama-sayfa). DR1'in durumu doğrulanmadı.
- **DESI Legacy Imaging Surveys.** Gezinme menüsünde DR11 görülmüştür (tarama-sayfa); yayım ayrıntısı okunmadı.
- **ZTF.** Açık veri sürümleri sayfası taramada 404 döndü; doğrulanmadı [Bellm2019].
- **Euclid.** Doğrulanmadı.

## Uzay teleskopları (MAST)

- **TESS.** Erişim: `astroquery.mast`, toplu indirme, TESSCut, bulut erişimi; MAST sayfasındaki son veri sürüm notu 12 Kasım 2026 tarihlidir (tarama-sayfa) [Ricker2015].
- **Kepler, HST, JWST:** doğrulanmadı.

## Ötegezegen ve küçük cisimler

- **NASA Exoplanet Archive.** Önerilen arayüz TAP'tır; eski `nstedAPI` arayüzünde tabloların çoğu kaldırılmıştır (tarama-sayfa).
- **JPL Horizons ve SBDB, Minor Planet Center:** doğrulanmadı (MPC sayfası taramada 403 döndü).

## Genel hizmetler

- **SIMBAD, VizieR, CDS X-Match, NED, IRSA, HEASARC, ESO arşivi, AAVSO, SkyView:** doğrulanmadı.
- **GWOSC** (çekimsel dalga açık verisi): taramada sertifika hatası; `gwosc.org` adresi denenecek.

## IVOA standartları

Tarama IVOA belge dizininden şu öneri sürümlerini okumuştur (tarama-sayfa): TAP 1.1 (27 Eylül 2019), ADQL 2.1 (15 Aralık 2023), SIA 2.0 (23 Aralık 2015), SSA 1.1 (10 Şubat 2012), Simple Cone Search 1.03 (22 Şubat 2008; 2.0 çalışma taslağı), VOTable 1.5 (16 Ocak 2025; 1.6 önerilen öneri aşamasında), UCD1+ 1.3 (27 Mayıs 2018), HiPS 1.0 (19 Mayıs 2017), MOC 2.0 (27 Temmuz 2022), DataLink 1.1 (15 Aralık 2023).

`arsiv/tap.py` ve `io/votable.py` TAP 1.1, ADQL 2.1 ve VOTable 1.5'i hedefler.

## Öncelik sırası

Yol haritasındaki projelere göre arşiv modülleri şu sırayla yazılır:

1. Pantheon+SH0ES (düz dosya indirme; TAP gerekmez).
2. Gaia TAP (astrometrik referans ve HR diyagramı).
3. MAST TESS ışık eğrileri.
4. SDSS tayfları.
5. JPL Horizons (efemeris kâhini).
6. NASA Exoplanet Archive, SIMBAD, VizieR.

İlk kullanımdan önce her arşivin uç noktası, veri sürümü ve atıf koşulu kendi belgesinden okunur ve modülün başına tarihiyle yazılır.
