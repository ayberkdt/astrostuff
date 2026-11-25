# Görüntü ve sinyal işleme matematiği

Hedef modüller: `indirgeme/`, `fotometri/`, `tayf/`, `zamanserisi/`.

Bir CCD karesi, gökyüzünün nokta dağılım fonksiyonuyla evrişip piksel ızgarasında örneklenmiş ve gürültü eklenmiş hâlidir. İndirgeme, fotometri ve tayf çıkarımının hemen her adımı bu üç işlemin (evrişim, örnekleme, gürültü) matematiğine dayanır. Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. Evrişim ve doğrusal süzgeçler

- Evrişim tanımı, değişme ve birleşme özelliği; evrişim teoremi: evrişimin Fourier dönüşümü dönüşümlerin çarpımıdır [VanderPlas2018, Böl. 2].
- Gözlenen görüntü: `G = (K ⊗ PSF) ⊗ piksel + gürültü`; piksel yanıtı da bir evrişimdir.
- Ayrılabilir çekirdekler: iki boyutlu Gauss evrişimi iki tek boyutlu evrişime ayrılır; yük `O(N k²)`'den `O(N k)`'ye iner.
- FFT ile evrişim: sıfır doldurma, sarma etkisi, kenar koşulları (yansıtma, sabit, maskeli normalleştirme).
- Maskeli evrişim: kötü pikseller ağırlık görüntüsüyle birlikte evriştirilir ve sonuç ağırlığın evrişimine bölünür.

## 2. Örnekleme ve enterpolasyon

- Örnekleme teoremi: bant sınırlı sinyal, en yüksek frekansının iki katında örneklenirse tam geri kurulur [Shannon1949].
- Görüntüde karşılığı: PSF'in yarı yükseklik genişliği yaklaşık iki pikselden darsa görüntü eksik örneklenmiştir; kaydırma ve enterpolasyon bilgi kaybeder, örtüşme oluşur.
- Enterpolasyon çekirdekleri: en yakın komşu, çift doğrusal, çift kübik, Lanczos-`a`. Her çekirdek bir alçak geçiren süzgeçtir: keskinlik, çınlama ve gürültü ilintisi arasında ödünleşim vardır.
- Yeniden örneklemede akı korunumu: Jacobian çarpanı; alan koruyan (piksel örtüşmesi) yöntemler.
- Eksik örneklenmiş karelerin alt piksel kaydırmalarla birleştirilmesi (drizzle): piksel küçültme katsayısı ve çıktı ızgarası; gürültü ilintisi [FruchterHook2002].
- Enterpolasyon gürültüyü ilintilendirir: komşu piksellerin kovaryansı sıfırdan farklı olur ve açıklık fotometrisinin belirsizliği köşegen varsayımıyla küçük çıkar. Düzeltme çarpanı çekirdekten hesaplanır.

## 3. Gürültü modelleri

- Poisson sayım gürültüsü ile Gauss okuma gürültüsünün toplamı; varyans `σ² = S/g + R²` (ADU biriminde, `g` kazanç) [Howell2006, Böl. 4; Janesick2007].
- Varyans kestiriminde yanlılık: varyansı gözlenen sayımdan kestirmek, düşük sayımlı pikselleri fazla ağırlıklandırır ve ağırlıklı ortalamayı aşağı çeker. Çözüm: varyansı modelden (düzgünleştirilmiş görüntüden) hesaplamak [Horne1986].
- Varyans dengeleyici dönüşüm: Poisson veride `2√(x + 3/8)` dönüşümü varyansı yaklaşık sabitler [Anscombe1948].
- Düşük sayım rejiminde `χ²` yerine Poisson olabilirliği [Cash1979].
- Düşük S/N'de kadir: logaritma eksi akıda tanımsızdır; asinh kadiri bu sorunu giderir [Lupton1999].

## 4. Saptama kuramı

- Eşleşmiş süzgeç: bilinen biçimli bir sinyal için beyaz Gauss gürültüsünde en yüksek S/N'yi veren doğrusal süzgeç sinyalin kendisidir. Nokta kaynak saptamada çekirdek PSF'tir [Stetson1987].
- Saptama eşiği ve yanlış pozitif oranı: `N` bağımsız çözünürlük elemanında `kσ` eşiğinin beklenen yanlış saptama sayısı; evrişimden sonra bağımsız eleman sayısı piksel sayısından küçüktür.
- Tamlık ve saflık eğrileri; alıcı işletim eğrisi.
- İlintili gürültüde eşleşmiş süzgeç: gürültü güç tayfıyla ağırlıklandırma (beyazlatma). Çekimsel dalga aramasının temel işlemidir [Allen2012].
- Görüntü farkında en iyi saptama istatistiği: iki görüntünün PSF ve gürültü düzeylerini birlikte hesaba katan fark [Zackay2016].

## 5. PSF modelleme ve uydurma

- Analitik profiller: Gauss, Moffat [Moffat1969], Airy; çok bileşenli toplamlar [King1971].
- Piksel üzerinde integral: profilin piksel merkezindeki değeri yerine piksel alanı üzerindeki integrali; dar PSF'te fark yüzde düzeyine çıkar.
- Deneysel (piksel tablolu) PSF: alt piksel ızgarasında yıldızlardan birleştirme; alan boyunca polinom değişim [Stetson1987].
- Akı ve konumun en büyük olabilirlik kestirimi; Cramér-Rao alt sınırı: astrometrik duyarlık yaklaşık `FWHM/(S/N)` mertebesindedir [King1971].
- PSF eşleme çekirdeği: iki görüntünün PSF'ini eşitleyen evrişim çekirdeğinin taban fonksiyonlarıyla doğrusal en küçük kareler çözümü [AlardLupton1998].

## 6. Ters problemler

- Ters evrişim kötü konumlanmıştır: gürültü yüksek frekanslarda büyür.
- Richardson-Lucy yinelemesi: Poisson olabilirliğinin beklenti-en büyükleme çözümü; akıyı ve artılığı korur, yineleme sayısı düzenlileştirme görevi görür [Richardson1972; Lucy1974].
- Wiener süzgeci: sinyal ve gürültü güç tayfları bilindiğinde en küçük ortalama kare hatalı doğrusal çözüm.
- İleri modelleme tercih edilir: görüntüyü ters çevirmek yerine model PSF ile evriştirilip veriyle karşılaştırılır.

## 7. Tayfsal çözümleme

- Ayrık Fourier dönüşümü, güç tayfı, Parseval eşitliği; frekans çözünürlüğü `1/T`.
- Pencere fonksiyonları ve sızıntı: dikdörtgen, Hann, Hamming, Blackman; ana lob genişliği ile yan lob düzeyi arasındaki ödünleşim [Harris1978].
- Eşit aralıklı olmayan örnekleme: gözlem penceresinin Fourier dönüşümü gerçek tayfla evrişir ve takma adlar üretir [VanderPlas2018, Böl. 3-4].
- Lomb-Scargle periyodogramı ve en küçük kareler yorumu [VanderPlas2018, Böl. 5-6]; yüzen ortalama ve Kepler biçimli genelleme [Zechmeister2009]; tepe anlamlılığı [Baluev2008]. Klasik ve yüzen ortalamalı biçimler `zamanserisi/periyodogram.py` içinde uygulanmıştır.
- Çapraz ilinti ve faz ilintisi: görüntü hizalama ve dikine hız ölçümü [TonryDavis1979].

## 8. Çok ölçekli yöntemler

- Dalgacık dönüşümü (à trous algoritması): ölçek ölçek ayrıştırma; yaygın kaynak ve arka plan ayrımı.
- Medyan ve morfolojik süzgeçler: arka plan haritası, kozmik ışın ve iz ayıklama [vanDokkum2001].
- Küre üzerinde harmonik çözümleme: küresel harmonikler ve HEALPix pikselleme [Gorski2005].

## 9. Kalibrasyonun doğrusal cebir olarak kurulması

- Çok geceli, örtüşen alanlı fotometrinin birlikte çözümü: yıldız kadirleri ve gece ya da kare sıfır noktaları tek bir seyrek doğrusal sistemin bilinmeyenleridir [Padmanabhan2008].
- Astrometrik blok düzeltmesi: bütün karelerin WCS parametreleri ve yıldız konumları birlikte uydurulur.
- Bu sistemler seyrek ve büyüktür; seyrek Cholesky ya da eşlenik gradyan ile çözülür.

## Doğrulama

- Evrişim: delta fonksiyonuyla evrişim çekirdeği verir; toplam akı korunur; FFT ve doğrudan toplam uyuşur.
- Enterpolasyon: kaydır ve geri kaydır deneyiyle çekirdek başına artık; ilintili gürültünün kovaryansı öngörülenle uyuşur.
- Eşleşmiş süzgeç: sentetik veride S/N kazancı kuramsal değere ulaşır; yanlış pozitif oranı eşikle öngörülen orandadır.
- Richardson-Lucy: bilinen sahnede akı korunumu ve yakınsama eğrisi.
- Periyodogram: `astropy.timeseries.LombScargle` ile `1e-10` düzeyinde uyum (uygulandı, `tests/kahin/test_periyodogram_kahin.py`).
