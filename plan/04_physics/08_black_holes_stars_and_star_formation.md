# Kara delikler, yıldız fiziği ve yıldız oluşumu: uygulanan modüller

Modüller: `karadelik/`, `yildiz/`, `yildizolusumu/`. Üçü de 25 Kasım 2026'da yazılmıştır. Fonksiyon başına kaynak ve teyit durumu [../DURUM.md](../DURUM.md) dosyasındadır; bu belge kapsamı, okunan kaynakları ve okuma sırasında bulunan uyuşmazlıkları kaydeder.

## Kara delikler (`karadelik`)

34 fonksiyon; 29'unun bütün kayıtları kaynak metninden okunmuştur.

**Kapsam.**

- Geometri: çekim yarıçapı ve Schwarzschild yarıçapı, foton küresi, kritik çarpma parametresi, gölgenin açısal çapı [EHT2019]; en içteki kararlı yörünge (ISCO) ve yörünge frekansı [Abramowicz2013].
- Kerr: ISCO, olay ufku, ergosfer, foton yörüngesi yarıçapları ve ufkun açısal hızı [Abramowicz2013; Page2005].
- Yığılma: Eddington ışınım gücü ve kütle akısı, Salpeter zamanı, Bondi yarıçapı ve akısı [Inayoshi2020]; ince disk sıcaklık profili [Armitage2011]; gelgit parçalanma yarıçapı ve Hills kütlesi [Komossa2015].
- Kuantum: Hawking sıcaklığı, Bekenstein-Hawking entropisi [Page2005].
- Çiftler: cıvıltı kütlesi, frekans türevi, birleşmeye kalan süre [Sathyaprakash2009].

**Okuma sırasında bulunanlar.**

- [Abramowicz2013] içindeki Kerr foton yörüngesi formülü basıldığı biçimiyle `a* = 1` için 4 çekim yarıçapı verir; bu geri yönlü yörüngedir. İleri yönlü yörünge için açının işareti değiştirilmiş ve sonuç kaynağın kendi kübik denklemine karşı sınanmıştır.
- Aynı kaynaktaki ISCO formülü yalnızca ileri yönlü yörünge içindir. Geri yönlü işaret [Bardeen1972] makalesinden gelir ve o makale okunmamıştır; kayıt "bekliyor" durumundadır. İki işaret de radyal episiklik koşula karşı sınanmıştır.
- [Abramowicz2013] Eddington ışınım gücünü Güneş kütlesi başına 1,2 × 10³⁸ erg/s verir; Thomson kesiti ve proton kütlesiyle hesaplanan değer 1,257 × 10³⁸'dir. Fark yuvarlamadır; test toleransı %6 alınmıştır.
- [EHT2019] değerleriyle M87* gölgesi 37,86 mikroyay saniyesi çıkar; makaledeki değer 37,6'dır.

**Bekleyen kayıtlar.** Çekimsel kırmızıya kayma [Carroll2019], ISCO özgül enerjisi ve verim bağıntısı [Bardeen1972], buharlaşma süresi [Page1976]. Bu fonksiyonlar bağımsız sınamalardan geçer (enerji en küçüğü, kareleme ile integral), ama kaynak metinleri açılmamıştır.

**Sınırlar.** Birleşen çiftlerde ISCO frekansı deneme parçacığı formülünün toplam kütleye uygulanmasıdır; kaba bir kestirimdir. Dalga biçimi, post-Newton terimleri ve dönen çiftler yoktur.

## Yıldız fiziği (`yildiz`)

39 fonksiyon; 34'ünün bütün kayıtları kaynak metninden okunmuştur.

**Kapsam.**

- Politroplar: Lane-Emden denkleminin sayısal çözümü, ilk sıfır, kütle katsayısı, merkez-ortalama yoğunluk oranı, `n = 0, 1, 5` kapalı çözümleri; kütle ve yarıçaptan merkez yoğunluğu, basıncı ve sıcaklığı; çekim enerjisi [Liu1995; Bludman2010].
- Ölçekleme: ışınım gücü ve etkin sıcaklık, yüzey çekimi [Prsa2016]; serbest düşme, Kelvin-Helmholtz ve nükleer zaman ölçekleri [McKee2007; Zinnecker2007]; Eddington sınırı ve kritik dönme [Paxton2013]; Eddington standart modeli [Bludman2010].
- Sıkışık son evreler: beyaz cüce durum denklemi katsayısı ve Chandrasekhar kütlesi [Bludman2010].
- Başlangıç kütle fonksiyonları: Salpeter, Kroupa, Chabrier; yoğunluk, ortalama kütle, kütle kesri ve örnekleme [Kroupa2001; Chabrier2003].

**Okuma sırasında bulunanlar.**

- [Bludman2010] Tablo I'deki kütle katsayısı sütunu yalnızca `n = 3` için standart tanımla (2,018) uyuşur; `n = 1,5` ve `n = 2` için farklı bir nicelik verir. Kod yalnızca ilk sıfır ve yoğunluk oranı sütunlarını kullanır.
- [Araujo2011] makalesindeki türev tablosu `n = 1` için yanlıştır; yalnızca `n = 3` Güneş değerleri test karşılaştırmasında kullanılmıştır.
- [Kroupa2001] parçaların yalnızca orantısını verir. Parçalar sürekli birleştirilmiştir; bu varsayımla makalede bildirilen ortalama kütle (0,36) ve sayı ile kütle payları yuvarlama içinde yeniden elde edilir.
- [Chabrier2003] tablo katsayıları 1 Güneş kütlesinde %0,5'lik bir sıçrama bırakır; yayımlanan sayılar korunmuştur.

**Yeniden üretilen kaynak değerleri.** `n = 3` Güneş modeli: merkez yoğunluğu 76,4 g/cm³, merkez basıncı 1,24 × 10¹⁷ dyn/cm², merkez sıcaklığı 1,2 × 10⁷ K. Chandrasekhar kütlesi 1,46 Güneş kütlesi (`μ_e = 2`). Güneş'in `log g` değeri 4,438068.

**Bekleyen kayıtlar.** Salpeter eğiminin özgün makalesi [Salpeter1955] (eğim değeri okunan iki makaleden alınmıştır) ve nükleer zaman ölçeği [Kippenhahn2012].

**Eksikler.** Görelilikli olmayan beyaz cüce kütle-yarıçap bağıntısının kaynaklı katsayısı, hidrostatik merkez basıncı alt sınırı, anakol ömrü ölçeklemesi. Tam yıldız yapısı çözümü ve izokronlar yol haritasının sonraki adımlarıdır ([03_yildiz_astrofizigi.md](03_yildiz_astrofizigi.md)).

## Yıldız oluşumu (`yildizolusumu`)

29 fonksiyon; hepsinin denklemi kaynak metninden okunmuştur.

**Kapsam.**

- Gaz: eşsıcaklık ses hızı, Mach sayısı, sayı ve kütle yoğunluğu, sütun ve yüzey yoğunluğu [Krumholz2015].
- Kararlılık: Jeans boyu ve kütlesi, Bonnor-Ebert kritik kütlesi ve yarıçapı, viryal parametre, manyetik kritik kütle [McKee2007; Krumholz2015].
- Zaman ölçekleri ve verim: serbest düşme ve geçiş süreleri, serbest düşme başına yıldız oluşum verimi, eşsıcaklık küre çöküşünün birikim hızı [McKee2007].
- Gökada ölçeği: Kennicutt-Schmidt yasası ve yörünge zamanı biçimi [Kennicutt1998], tüketme zamanı, gaz diski için Toomre Q, ışınım gücünden yıldız oluşum hızı kalibrasyonları [Kennicutt2012].

**Okuma sırasında bulunanlar.**

- Jeans kütlesinin tanımı kaynaktan kaynağa değişir: [McKee2007] Jeans boyunun yarısı yarıçaplı kürenin kütlesini, [Krumholz2015] Jeans boyunun yarısı kenarlı küpün kütlesini alır. Oran 4π/3'tür; fonksiyon ikisini de sunar.
- Manyetik kritik kütlenin katsayısı için [McKee2007] 0,17, [Krumholz2015] 0,12 verir (farklı sayısal çalışmalardan). Varsayılan 0,17'dir.
- [Kennicutt2012] Tablo 1 kalibrasyonları, aynı Hα ışınım gücü için [Kennicutt1998] değerinin 0,68 katı yıldız oluşum hızı verir (başlangıç kütle fonksiyonu farkı).
- Bonnor-Ebert katsayıları (1,182; yoğunluk karşıtlığı 14,04) denklemin sayısal çözümünden yeniden elde edilmiştir.

**Eksikler.** Isıl ve türbülans desteğini birleştiren etkin hız dağılımı, yıldız diski için Toomre Q, tek tek gökadalara uygulanan yasaların gözlemle karşılaştırılması.

## Sınama yaklaşımı

Bu üç modülün bağımsız bir kitaplık kâhini yoktur. Sınama dört kaynaktan gelir:

- Kapalı çözümler ve limitler (Lane-Emden `n = 0, 1, 5`; düzgün küre; dönmeyen kara delik).
- Kaynaklarda basılı sayılar (tablolar, örnek değerler), toleransı basılan basamak sayısına bağlı.
- Bağımsız hesap yolları: aynı niceliğin başka bir denklemden ya da sayısal integralden elde edilmesi (disk ışınım gücü, birleşme süresi, Salpeter büyümesi).
- `hypothesis` ile değişmezler: tekdüzelik, ölçekleme, gidiş-dönüş.

Galeri şekilleri (`examples/physics_gallery.py`) bu fonksiyonların doğrudan çıktısıdır.
