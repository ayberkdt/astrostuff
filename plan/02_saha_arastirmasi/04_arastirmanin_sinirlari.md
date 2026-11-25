# Araştırmanın yöntemi ve sınırları

Planın hangi bölümüne ne kadar güvenilebileceği bilginin nasıl elde edildiğine bağlıdır. Dört güven düzeyi vardır.

## 1. Kaynağından okunan ve kaydı tutulan

- **Künyeler.** `kaynakca/kaynaklar.bib` içindeki her giriş `dogrula.py` ile dış kayda karşı sınanmıştır: DOI'ler Crossref'e, arXiv kimlikleri arXiv API'sine, ISBN'ler Open Library'ye. Sınama başlık benzerliğini ve yılı karşılaştırır; sonuç `dogrulama_kaydi.json` dosyasındadır. Bu sınama künyenin var olduğunu ve başlığın, yılın, derginin doğru yazıldığını gösterir; kaynağın içeriğinin planda doğru aktarıldığını göstermez.
- **Paket sürümleri.** `paketler.txt` içindeki 100 paketin sürümü ve yükleme tarihi PyPI JSON API'sinden okunmuştur.
- **Pantheon+SH0ES veri sürümü.** Dosya listesi, satır sayısı, sütun adları, kovaryans boyutu ve lisans durumu GitHub'dan okunmuştur.
- **Paket adı.** `kozmos` adının PyPI'da boş olduğu sorgulanmıştır.

## 2. Web taramasının sayfayı açarak bildirdiği

Altı kollu bir tarama (çekirdek ekosistem, konum astronomisi, veri indirgeme, kozmoloji, istatistik ve arşivler, astrofizik alan kodları) yapılmıştır. Tarama her bilgi için okuduğu adresi bildirmiştir. Bakım durumları, arşiv sürümleri, IVOA standart sürümleri ve güncel `H0` değerleri bu düzeydedir. Bu bilgiler ikinci kez doğrulanmamıştır; belgelerde "taramaya göre" ya da "tarama-sayfa" diye işaretlidir.

## 3. Taramanın yalnızca arama özetinde gördüğü

Union3, DES-SN5YR yeniden kalibrasyonu, CosmoVerse raporu, SH0ES HST+JWST birleşik değeri ve JWST TRGB/JAGB değerleri bu düzeydedir ("tarama-özet"). Kullanılmadan önce kaynağından okunmalıdır.

## 4. Hatırlanan bilgi

Matematik, fizik ve veri bilimi bölümlerindeki denklemler, algoritma adımları, sayısal katsayılar, denklem ve bölüm numaraları, tipik büyüklükler ve yöntemlerin birbirine göre üstünlükleri planı yazanın bilgisinden gelir; bu çalışmada kaynak metinleri açılıp okunmamıştır. Atıflar denklemin hangi kaynakta aranacağını gösterir, denklemin o kaynaktan aktarıldığını göstermez.

Bu nedenle bir fonksiyon yazılmadan önce atıf verilen kaynağın ilgili bölümü açılır, denklem oradan alınır ve `@kaynak` kaydına denklem numarası o okumadan yazılır. Plandaki denklem ile kaynaktaki farklıysa kaynak geçerlidir ve plan düzeltilir.

## Taramanın ulaşamadıkları

- Yayıncı sayfalarının bir kısmı (A&A, IERS, USNO, ESO, MPC, GWOSC, Euclid, ZTF, Planck Legacy Archive) otomatik erişimi reddetti ya da boş döndü.
- PyPI'nin HTML sayfaları sık sık yüklenemedi; sürümler bu yüzden JSON API'sinden yeniden okundu.
- Astrofizik alan kodları kolunun büyük kısmı tamamlanamadı; ders kitaplarının künyeleri taramadan değil, Crossref ve Open Library sınamasından gelir.
- Taramanın bellekten denediği birkaç arXiv kimliği ilgisiz makalelere çıktı ve atıldı. Kimlik ve DOI'nin bellekten yazılmaması, yazılanın sınanması gerektiğinin göstergesidir.

## Künyesi eksik kaynaklar

Belgelerde "künyesi kaynakçaya eklenecek" diye geçen konular: galaktik koordinat sisteminin tanımı (Hipparcos kataloğu), SIP bozulma kuralı, HEALPix, Cash istatistiği, NUTS örnekleyici, VSOP87 ve ELP kuramları, atmosferik türbülans ve görüş, Ay'ın gökyüzü parlaklığı modeli, Moffat profili, hava-boşluk dalga boyu dönüşümü, Landy-Szalay kestiricisi, FITS 4.0 belgesi. Bunlar ilgili modül yazılmadan önce kaynakçaya eklenir ve `dogrula.py` ile sınanır.

DOI'si olmayan dört kaynağın künyesi elle doğrulanacaktır: [LeavittPickering1912], [Tripp1998], [TullyFisher1977], [SOFA2023]. [IERS2010] künyesinin ISBN'i Open Library'de bulunamamıştır.

## Yeniden doğrulama

```
python plan/kaynakca/dogrula.py
```

Betik üç denetimi yeniden koşar ve kaydı günceller. Paket sürümleri ve arşiv durumları zamanla değişir; bu belgelerdeki tarihli bilgiler yeni bir koşuyla yenilenmeden güncel sayılmaz.
