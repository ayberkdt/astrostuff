# Araştırmanın yöntemi ve sınırları

Planın hangi bölümüne ne kadar güvenilebileceği bilginin nasıl elde edildiğine bağlıdır. Dört güven düzeyi vardır.

## 1. Kaynağından okunan ve kaydı tutulan

- **Künyeler.** `bibliography/references.bib` içindeki her giriş `dogrula.py` ile dış kayda karşı sınanmıştır: DOI'ler Crossref'e, arXiv kimlikleri arXiv API'sine, ISBN'ler Open Library'ye. Sınama başlık benzerliğini ve yılı karşılaştırır; sonuç `dogrulama_kaydi.json` dosyasındadır. Bu sınama künyenin var olduğunu ve başlığın, yılın, derginin doğru yazıldığını gösterir; kaynağın içeriğinin planda doğru aktarıldığını göstermez.
- **Paket sürümleri.** `paketler.txt` içindeki 100 paketin sürümü ve yükleme tarihi PyPI JSON API'sinden okunmuştur.
- **Pantheon+SH0ES veri sürümü.** Dosya listesi, satır sayısı, sütun adları, kovaryans boyutu ve lisans durumu GitHub'dan okunmuştur.
- **Paket adı.** `kozmos` adının PyPI'da boş olduğu sorgulanmıştır.
- **Uygulanan fonksiyonların denklemleri.** Aşağıdaki kaynakların metni açılmış ve kodlanan denklemler oradan alınmıştır: Hogg 1999, Brout ve ark. 2022, Hogg, Bovy ve Lang 2010, Foreman-Mackey ve ark. 2013, VanderPlas 2018, Prša ve ark. 2016 ve Mamajek ve ark. 2015 (arXiv TeX kaynakları); CODATA 2022 (NIST `allascii.txt`); ERFA kaynak dosyaları (`cal2jd.c`, `jd2cal.c`, `era00.c`, `gmst06.c`, `obl06.c`, `sepp.c`, `s2c.c`, `c2s.c`, `anp.c`, `anpm.c`, `icrs2g.c`, `hd2ae.c`, `ae2hd.c`, `erfam.h`); Pantheon+SH0ES veri deposunun `README` dosyası. Fonksiyon başına durum [../DURUM.md](../DURUM.md) dosyasındadır.

## 2. Web taramasının sayfayı açarak bildirdiği

Altı kollu bir tarama (çekirdek ekosistem, konum astronomisi, veri indirgeme, kozmoloji, istatistik ve arşivler, astrofizik alan kodları) yapılmıştır. Tarama her bilgi için okuduğu adresi bildirmiştir. Bakım durumları, arşiv sürümleri, IVOA standart sürümleri ve güncel `H0` değerleri bu düzeydedir. Bu bilgiler ikinci kez doğrulanmamıştır; figuresde "taramaya göre" ya da "tarama-sayfa" diye işaretlidir.

## 3. Taramanın yalnızca arama özetinde gördüğü

Union3, DES-SN5YR yeniden kalibrasyonu, CosmoVerse raporu, SH0ES HST+JWST birleşik değeri ve JWST TRGB/JAGB değerleri bu düzeydedir ("tarama-özet"). Kullanılmadan önce kaynağından okunmalıdır.

## 4. Hatırlanan bilgi

Matematik, fizik ve veri bilimi bölümlerindeki denklemler, algoritma adımları, sayısal katsayılar, denklem ve bölüm numaraları, tipik büyüklükler ve yöntemlerin birbirine göre üstünlükleri planı yazanın bilgisinden gelir; yukarıda sayılan kaynaklar dışında kaynak metinleri açılıp okunmamıştır. Atıflar denklemin hangi kaynakta aranacağını gösterir, denklemin o kaynaktan aktarıldığını göstermez.

Bu nedenle bir fonksiyon yazılmadan önce atıf verilen kaynağın ilgili bölümü açılır, denklem oradan alınır ve `@kaynak` kaydına denklem numarası o okumadan yazılır. Plandaki denklem ile kaynaktaki farklıysa kaynak geçerlidir ve plan düzeltilir.

## Taramanın ulaşamadıkları

- Yayıncı sayfalarının bir kısmı (A&A, IERS, USNO, ESO, MPC, GWOSC, Euclid, ZTF, Planck Legacy Archive) otomatik erişimi reddetti ya da boş döndü.
- PyPI'nin HTML sayfaları sık sık yüklenemedi; sürümler bu yüzden JSON API'sinden yeniden okundu.
- Astrofizik alan kodları kolunun büyük kısmı tamamlanamadı; ders kitaplarının künyeleri taramadan değil, Crossref ve Open Library sınamasından gelir.
- Taramanın bellekten denediği birkaç arXiv kimliği ilgisiz makalelere çıktı ve atıldı. Kimlik ve DOI'nin bellekten yazılmaması, yazılanın sınanması gerektiğinin göstergesidir.

## Künyesi eksik kaynaklar

İlk taramada künyesi eksik kalan konuların çoğu sonradan kaynakçaya eklenmiş ve sınanmıştır (Hipparcos kataloğu, SIP kuralı, HEALPix, Cash istatistiği, NUTS, VSOP87, görüş, Ay parlaklığı modeli, Moffat profili, havanın kırılma indisi, Landy-Szalay kestiricisi, FITS 4.0). ELP Ay kuramının künyesi hâlâ eksiktir.

DOI, arXiv kimliği ya da ISBN taşımayan kaynakların künyesi elle doğrulanacaktır; güncel liste `bibliography/VERIFICATION_SUMMARY.md` dosyasının "Elle doğrulanacak" bölümündedir. [IERS2010] künyesinin ISBN'i Open Library'de bulunamamıştır.

## Yeniden doğrulama

```
python bibliography/verify.py
```

Betik üç denetimi yeniden koşar ve kaydı günceller. Paket sürümleri ve arşiv durumları zamanla değişir; bu figuresdeki tarihli bilgiler yeni bir koşuyla yenilenmeden güncel sayılmaz.
