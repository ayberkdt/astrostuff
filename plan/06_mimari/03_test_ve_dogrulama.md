# Test ve doğrulama

Bir fonksiyon dört bağımsız sınamadan geçince doğrulanmış sayılır. Sınamalar farklı hata türlerini yakalar; biri diğerinin yerini tutmaz.

## Dört sınama

**1. Kaynağın çalışılmış örneği (`tests/birim`).** Kaynakta sayısal örnek varsa test o örneği yeniden üretir ve kaynağı sayfasıyla anar. Meeus'un kitabındaki numaralı örnekler [Meeus1998], SOFA yemek kitaplarındaki örnek çıktılar [SOFA2023] ve Hogg'un şekilleri [Hogg1999] bu türün başlıca kaynaklarıdır. Yakaladığı hata: denklemin yanlış aktarılması.

**2. Kâhin karşılaştırması (`tests/kahin`).** Aynı girdi bağımsız bir uygulamaya verilir ve fark toleransın altında olmalıdır. Kâhinler: koordinat ve zaman için ERFA ve Astropy, efemeris için Skyfield ve JPL Horizons, kozmoloji için `astropy.cosmology`, fotometri için photutils ve sep, periyodogram için `astropy.timeseries`, örnekleme için emcee. Girdiler tanım kümesini tarayan sabit tohumlu rastgele noktalar ve elle seçilmiş sınır durumlarıdır. Yakaladığı hata: örnekte görünmeyen bölgelerdeki sapma.

Kâhin de yanılabilir. Fark çıktığında önce kaynağa dönülür; kâhin farklı bir model ya da sabit kullanıyorsa fark belgelenir ve tolerans buna göre konur.

**3. Özellik testi (`tests/ozellik`).** `hypothesis` ile üretilen girdilerde değişmezler sınanır:

- gidiş-dönüş: `galaktik_den_ekvatoral(ekvatoral_den_galaktik(x)) = x`
- korunum: dönme matrisleri ortogonaldir, determinant +1'dir; iki cisim integrasyonunda enerji ve açısal momentum korunur
- limit: `z → 0` için `D_L → cz/H0`; `e → 0` için eksantrik anomali ortalama anomaliye gider
- simetri ve tekdüzelik: hava kütlesi zenit açısıyla artar; eşyürüyen uzaklık `z` ile artar
- boyut: çıktının birimi girdilerin birimlerinden beklenen birimdir

Yakaladığı hata: kutup, sıfır geçişi, 0/2π sarması gibi sınır durumları.

**4. Uçtan uca bilim testi (`projeler`).** Bir analiz bilinen bir sonucu yeniden üretir: Pantheon+ verisinden yayımlanan `H0` ve `Ωm` aralığı, bilinen bir ötegezegenin dönemi ve geçiş derinliği, bir açık kümenin yayımlanmış uzaklık modülü. Yakaladığı hata: tek tek doğru fonksiyonların yanlış birleştirilmesi.

## Tolerans

Tolerans her test için ayrı yazılır ve gerekçesi yorum satırında durur. Başlangıç değerleri:

- Kapalı biçimli dönüşümler (küresel trigonometri, birim dönüşümü): bağıl `1e-12`.
- Seri açılımına dayanan modeller (presesyon, nutasyon): kâhinle aynı model kullanılıyorsa `1e-10` rad; farklı modelde kaynakların bildirdiği model farkı.
- Sayısal integral ve kök bulma: istenen toleransın on katı.
- Gürültülü veriyle çalışan algoritmalar (fotometri, periyodogram): sentetik veride gerçeğe karşı, beklenen istatistik belirsizliğin katı olarak.

Bu değerler başlangıç önerisidir; ilk testler yazılırken ölçülen farkla yeniden konur ve değişiklik kaydedilir.

## Sentetik veri

Gözlem verisi işleyen her modülün yanında bir üreteç bulunur: bilinen parametrelerle yapay CCD görüntüsü (yıldız alanı, PSF, gökyüzü arka planı, Poisson ve okuma gürültüsü, bias, karanlık akım, düz alan, kozmik ışın), yapay tayf ve yapay ışık eğrisi. Algoritma önce gerçeğin bilindiği sentetik veride, sonra gerçek veride sınanır. Enjeksiyon-geri kazanım deneyleri (bilinen sinyali gerçek veriye ekleyip geri bulma) tamlık ve yanlılık ölçümünün standart aracıdır.

## Referans veri

`veri/referans` altında küçük, sürüm denetimindeki dosyalar durur: kâhin çıktıları (hangi kütüphanenin hangi sürümüyle üretildiği başlıkta yazılı), kaynaklardan aktarılmış örnek tabloları ve küçük gerçek veri kesitleri. Büyük veri indirilir ve `veri/ham` altında, sürüm denetimi dışında, SHA-256 özetiyle kayıtlı tutulur.

## Sürekli denetim

Her değişiklikte: `ruff`, `mypy --strict`, birim ve özellik testleri, kaynak kuralı testi. Kâhin testleri `dogrulama` bağımlılıklarını gerektirdiğinden ayrı işaretlenir (`pytest -m kahin`) ve en az birleştirme öncesinde koşulur. Ağ gerektiren testler `-m ag` ile ayrılır ve varsayılan koşuda çalışmaz.
