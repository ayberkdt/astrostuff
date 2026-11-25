# Kod standartları ve kaynak gösterme

Her açık fonksiyon, uyguladığı denklemin ya da algoritmanın yayımlanmış kaynağına bağlanır. Bağ üç yerde kurulur ve üçü de makineyle denetlenir: belge dizgisi, `@kaynak` dekoratörü ve test.

## Kaynak gösterme sistemi

### Tek kaynakça dosyası

Bütün kaynaklar `kaynakca/kaynaklar.bib` dosyasında BibTeX biçiminde durur. Anahtar biçimi `İlkYazarSoyadıYıl` (`Hogg1999`), aynı yazar ve yılda `Hogg1999a`; kurum belgelerinde kısa ad ve yıl (`IERS2010`, `SOFA2021`). Kod, belge ve test bu anahtarı kullanır; başka yerde künye yazılmaz.

Her girişte `doi`, yoksa `eprint` (arXiv) ya da `isbn` bulunur. `kaynakca/dogrula.py` DOI'leri Crossref'ten sorgular, başlığı ve yılı karşılaştırır ve sonucu `dogrulama_kaydi.json` dosyasına yazar. Kayıtta doğrulanmamış görünen bir giriş kodda kullanılabilir, ama künyesi doğrulanmış sayılmaz.

### Belge dizgisi

NumPy belge dizgisi biçimi kullanılır. `Kaynaklar` bölümü zorunludur ve yalnızca anahtar değil, kaynağın içindeki yeri de verir: denklem numarası, bölüm, sayfa ya da tablo.

```python
@kaynak("Hogg1999", yer="Denk. 15", teyit="okundu")
def esyuruyen_uzaklik_mpc(z, kozmoloji):
    """Bakış doğrultusundaki eşyürüyen uzaklığı hesaplar.

    D_C = D_H * ∫_0^z dz' / E(z'),  D_H = c / H0

    Parametreler
    ------------
    z : float ya da ndarray
        Kırmızıya kayma, z >= 0.
    kozmoloji : Kozmoloji
        H0 [km s^-1 Mpc^-1] ve yoğunluk parametreleri.

    Döndürür
    --------
    float ya da ndarray
        Eşyürüyen uzaklık [Mpc].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 15.

    Geçerlilik
    ----------
    Radyasyon terimi E(z) içindedir; z > 1e4 için integral çözünürlüğü sınanmamıştır.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji.py: astropy.cosmology.FlatLambdaCDM ile bağıl fark < 1e-8.
    """
```

Bölümlerin sırası sabittir: tek satırlık özet, denklem, `Parametreler`, `Döndürür`, `Kaynaklar`, `Geçerlilik`, `Doğrulama`. `Geçerlilik` yaklaşımın sınırını yazar (kesilen seri terimi, geçerli zaman aralığı, beklenen doğruluk); sınır bilinmiyorsa "sınanmadı" yazılır.

### `@kaynak` dekoratörü

`kozmos/_kaynak.py` içindeki dekoratör fonksiyonu değiştirmez; yalnızca `(modül, fonksiyon, anahtar, yer, teyit, not)` kaydını bir kütüğe ekler. `yer` kaynağın içindeki konumdur (denklem numarası, bölüm, tablo, dosya adı ya da TeX etiketi). `teyit` kaydın kanıt statüsüdür: `okundu` denklemin kod yazılırken kaynağın metninden alındığını, `bekliyor` kaynağın henüz açılmadığını bildirir. Kütükten üç şey üretilir:

- `kozmos.kaynaklar(fonksiyon)`: fonksiyonun dayandığı künyeleri döndürür.
- `kozmos.kaynakca_raporu()`: bir analizde çağrılan fonksiyonların kaynak listesini verir; makale yazarken atıf listesi buradan çıkar.
- Kaynak → fonksiyon ters dizini: bir kaynakta hata bulunursa etkilenen fonksiyonlar listelenir.

### Makine denetimi

`tests/test_kaynak_kurali.py` aşağıdakileri sınar ve herhangi biri bozulduğunda test başarısız olur:

1. `kozmos` altında adı alt çizgiyle başlamayan her fonksiyon ve sınıf en az bir `@kaynak` taşır.
2. Kullanılan her anahtar `kaynaklar.bib` içinde vardır.
3. Belge dizgisindeki `Kaynaklar` bölümü dekoratördeki anahtarlarla aynı kümeyi içerir.
4. Her `@kaynak` kaydında `yer` doludur ve `teyit` iki geçerli değerden biridir.
5. Her açık fonksiyonun `Doğrulama` bölümünde adı geçen test dosyası vardır.

Kaynağı olmayan yardımcılar (dizi biçimlendirme, argüman denetimi) alt çizgiyle adlandırılır ve kuralın dışında kalır. Açık kalması gereken ama denklem uygulamayan fonksiyonlar `@altyapi("gerekçe")` ile işaretlenir; test bu muafiyetlerin gerekçeli ve en çok beş tane olduğunu denetler. Fonksiyon bir kaynağa değil bu projedeki bir türetime dayanıyorsa türetim `plan/03_matematik` ya da `docs/turetimler` altında yazılır ve `@kaynak("kozmos:turetim/<ad>")` ile gösterilir.

### Sabitler

Fiziksel ve astronomik sabitler `temel/sabitler.py` dosyasında, her biri değeri, birimi, belirsizliği ve kaynağıyla tanımlanır. Fiziksel sabitler CODATA önerilen değerlerinden, Güneş ve gezegen nominal değerleri IAU 2015 B3 kararından [Prsa2016], astronomik birim IAU 2012 B2 kararından alınır. Kod içinde çıplak sayı sabiti yazılmaz; `ruff` kuralı ve gözden geçirme bunu denetler.

## Python standartları

- Python 3.12 ve üstü; tip ipuçları zorunlu, `mypy --strict` temiz.
- Biçim ve denetim `ruff` ile; yapılandırma `pyproject.toml` içinde.
- Bir modül bir konuyu taşır; 400 satırı aşan modülün bölünmesi gözden geçirilir.
- Fonksiyon bir iş yapar. Bayrak argümanıyla davranış değiştiren fonksiyon ikiye ayrılır.
- Konumsal argüman en çok üç tanedir; fazlası anahtar sözcükle verilir (`*` ayırıcı).
- Hata sessizce geçilmez: geçersiz girdi `ValueError` üretir, tanım kümesi dışındaki dizi elemanı için davranış (`nan` ya da hata) belge dizgisinde yazılıdır.
- Değişebilir varsayılan argüman, küresel değişken ve içe aktarma sırasında yan etki yasaktır.
- Rastgelelik `numpy.random.Generator` nesnesi argüman olarak alınarak kullanılır; küresel tohum kullanılmaz.
- Açılar içeride radyandır. Derece, saat ve yay saniyesi yalnızca `temel/aci.py` dönüşümleriyle sınırda görünür.
- Dizi fonksiyonları `numpy.asarray` ile girdi alır, skaler girdide skaler döndürür.

## Adlandırma

- Fonksiyon adı hesaplanan niceliği ve birimini söyler: `isiklilik_uzakligi_mpc`, `hava_kutlesi`, `julyen_tarihi`.
- Dönüşüm fonksiyonları `kaynak_den_hedef` biçimindedir: `ekvatoral_den_galaktik`, `utc_den_tt`.
- Semboller kaynaktaki gösterimi izler ve belge dizgisinde açıklanır; iki kaynak farklı sembol kullanıyorsa `@kaynak` notunda eşleme yazılır.

## Sürüm ve değişiklik

Sayısal sonucu değiştiren her değişiklik (yeni seri terimi, sabit güncellemesi, farklı integral yöntemi) `DEGISIKLIKLER.md` dosyasına, etkilenen fonksiyon ve değişimin büyüklüğüyle yazılır. Kâhin testlerinin toleransı gevşetilecekse gerekçe aynı kayıtta durur.
