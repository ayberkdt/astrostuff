# Görüntü indirgeme hattı

Hedef modül: `indirgeme/`. Fiziksel model: [../04_fizik/06_dedektor_ve_atmosfer.md](../04_fizik/06_dedektor_ve_atmosfer.md).

Ham dedektör çıktısı bilim görüntüsüne aşağıdaki sırayla dönüştürülür. Her adım bir fonksiyondur: veri, belirsizlik ve maske alır; aynı üçlüyü ve işlem kaydını döndürür.

## 1. Veri modeli

- Görüntü kabı: `veri` (2B dizi), `varyans`, `maske` (bit alanı: kötü piksel, doymuş, kozmik ışın, kenar, enterpole), `baslik`, `gecmis`.
- Birim izlenir: ADU, elektron, elektron/saniye. Kazanç uygulanmadan Poisson varyansı hesaplanmaz.
- Varyans yayılımı her adımda açıkça yazılır: çıkarmada toplanır, bölmede göreli varyanslar toplanır [Howell2006, Böl. 4].

## 2. Kalibrasyon karelerinin hazırlanması

- **Ana bias:** çok sayıda sıfır süreli karenin sigma kırpmalı ortalaması ya da medyanı. Aşırı tarama bölgesi varsa satır bazında düzey düzeltmesi önce yapılır.
- **Ana karanlık:** bias çıkarılmış karanlık karelerin birleşimi; poz süresine ölçekleme yalnızca karanlık akım doğrusal ise geçerlidir. Sıcak piksel haritası buradan çıkar.
- **Ana düz alan:** filtre başına; bias ve karanlık çıkarılmış düz karelerin, her biri kendi medyanına normalleştirildikten sonra birleşimi. Alacakaranlık düzlerinde yıldızlar kaydırma ve medyanla ayıklanır.
- **Kötü piksel maskesi:** düz alan oranlarından (iki farklı aydınlatma düzeyi) ve karanlık karelerden.
- Birleştirme yöntemleri: ortalama, medyan, sigma kırpmalı ortalama, en düşük/en yüksek atma; sağlam ölçek kestirimi [Beers1990]. Medyanın varyansı Gauss gürültüsünde ortalamanınkinin yaklaşık `π/2` katıdır; seçim buna göre yapılır.
- Kalibrasyon kalitesi ölçülür: ana karelerin gürültüsü bilim karesinin gürültüsüne eklenir; gereken kare sayısı buradan hesaplanır.

## 3. Bilim karesinin indirgenmesi

```
indirgenmiş = ( ham − aşırı_tarama − bias − karanlık · (t/t_k) ) / düz_alan
```

Sıra:

1. Aşırı tarama düzeltmesi ve kırpma.
2. Doğrusallık düzeltmesi (aygıt gerektiriyorsa) ve doymuş piksellerin maskelenmesi.
3. Bias çıkarma.
4. Karanlık çıkarma.
5. Kazanç uygulama (elektrona çevirme) ve varyans düzleminin kurulması: `σ² = sinyal + okuma_gürültüsü²`.
6. Düz alana bölme.
7. Saçak düzeltmesi (kırmızı bantlarda ince dedektörlerde).
8. Kötü piksel maskesinin uygulanması.

Foton aktarım eğrisiyle kazanç ve okuma gürültüsü ölçümü ayrı bir araçtır ve aygıt tanım dosyasını üretir [Janesick2007].

## 4. Kozmik ışın temizliği

- **Tek karede:** Laplace kenar saptama. Görüntü alt örneklenir, Laplace çekirdeğiyle evriştirilir, gürültü modeline göre anlamlı ve çevresine göre keskin pikseller işaretlenir; yıldızlardan "ince yapı" görüntüsüyle ayrılır [vanDokkum2001]. Kâhin: `astroscrappy`.
- **Çok karede:** hizalanmış karelerin sigma kırpmalı birleşimi; daha güvenilirdir.
- Uydu ve uçak izleri: Hough dönüşümü ya da doğrusal yapı saptama; maskeleme.

Temizlenen pikseller enterpole edilir ve maskede işaretlenir; fotometri bu pikselleri ağırlıklandırmada bilir.

## 5. Arka plan kestirimi

- Küresel: sigma kırpmalı medyan ya da kip kestirimi (`kip ≈ 2,5 · medyan − 1,5 · ortalama`) [BertinArnouts1996].
- Yerel harita: görüntü ağ hücrelerine bölünür, her hücrede kırpmalı kestirim, hücreler arası medyan süzgeci ve spline enterpolasyonu [BertinArnouts1996]. Kâhin: `sep` [Barbary2016] ve `photutils`.
- Kaynak maskeleme ve yineleme: kaynaklar saptanır, maskelenir, arka plan yeniden kestirilir.
- Arka plan gürültü haritası (RMS) saptama eşiğinin ve fotometri belirsizliğinin girdisidir.
- Hücre boyutu seçimi: kaynaklardan büyük, arka plan değişim ölçeğinden küçük. Geniş kaynaklarda (gökada, bulutsu) aşırı çıkarma riski belgelenir.

## 6. Hizalama ve birleştirme

- Kaydırma kestirimi: yıldız eşleme (üçgen ya da asterizm değişmezleri), faz ilintisi, WCS tabanlı.
- Yeniden örnekleme: enterpolasyon çekirdekleri ve akı korunumu; gürültü ilintisi doğar ve varyans haritasına yansıtılır.
- Birleştirme: ağırlıklı ortalama (`1/σ²` ya da S/N² ağırlığı), sigma kırpma, ölçek ve sıfır noktası eşitleme.
- Eksik örneklenmiş verinin kaydırmalı karelerden yeniden kurulması (drizzle) [FruchterHook2002].
- Görüntü farkı: PSF eşleme çekirdeği [AlardLupton1998] ve istatistiksel olarak en iyi fark görüntüsü [Zackay2016]. Geçici kaynak aramasının temelidir; ileri aşama.

## 7. Kalite denetimi

Her gece için otomatik rapor: bias düzeyi ve okuma gürültüsünün zamanla seyri, düz alan düzgünlüğü, görüş (FWHM) ve eliptiklik, gökyüzü parlaklığı, sıfır noktası ve geçirgenlik, limit kadiri, astrometrik artıklar. Eşik dışı kareler işaretlenir, sessizce atılmaz.

## 8. Sentetik veriyle doğrulama

Üreteç bilinen parametrelerle ham kare üretir: yıldız alanı (konum, akı), PSF, gökyüzü, Poisson gürültüsü, düz alan deseni, karanlık akım ve sıcak pikseller, bias yapısı, okuma gürültüsü, kozmik ışınlar, kazanç. İndirgeme hattı bu kareyi işler; sonuç bilinen gerçekle karşılaştırılır:

- İndirgenmiş görüntünün artığı gürültü modelinin öngördüğü dağılımdadır (indirgenmiş `χ²` yaklaşık 1).
- Kozmik ışın saptamanın tamlığı ve yanlış pozitif oranı ölçülür.
- Varyans düzlemi gerçek saçılmayı öngörür.

Gerçek veride ikinci sınama `ccdproc` ile aynı ham karelerin indirgenmesi ve piksel düzeyinde karşılaştırmadır.
