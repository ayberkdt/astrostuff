# Proje: Hubble diyagramı ve Hubble sabiti

Projenin ilk uçtan uca bilim hedefidir: açık veriden Hubble diyagramını kurmak, kozmolojik parametreleri kestirmek ve yayımlanmış sonucu yeniden üretmek. Fizik: [../04_fizik/05_kozmoloji.md](../04_fizik/05_kozmoloji.md). Veri: [../02_saha_arastirmasi/02_hubble_ve_kozmoloji_verisi.md](../02_saha_arastirmasi/02_hubble_ve_kozmoloji_verisi.md).

Proje basamaklara ayrılmıştır; her basamak bir öncekinin kodunu kullanır ve kendi kabul ölçütü vardır.

## Durum

Basamak 1, 2 ve 3 uygulanmış ve 25 Kasım 2026'da koşulmuştur (`projects/hubble/hubble_analysis.py`, kayıt `projects/hubble/result.json`):

- Basamak 2, yalnızca süpernovalar (N = 1580, `zHD > 0,01`, kalibratörler hariç), düz ΛCDM: Ωm = 0,332 ± 0,018; χ² = 1387,1, serbestlik derecesi 1578. Yayımlanan değer 0,334 ± 0,018 [Brout2022]; fark −0,07σ.
- Basamak 3, Sefeid kalibratörleriyle (N = 1657, 77 kalibratör satırı): H0 = 73,53 ± 1,00 km s⁻¹ Mpc⁻¹, Ωm = 0,331 ± 0,018, M = −19,244 ± 0,029. Yayımlanan değer 73,6 ± 1,1 [Brout2022]; fark −0,05σ.
- Zincir: 32 yürüyücü, 2000 adım, öz ilinti süresi en çok 36 adım, etkin örnek sayısı yaklaşık 1600, bölünmüş R̂ en çok 1,03.

Yayımlanan değerler makalenin arXiv TeX kaynağındaki makrolardan okunmuştur. Bulunan `H0` belirsizliği yayımlanandan küçüktür (1,00 ve 1,1); nedeni incelenmemiştir. Basamak 0 (Hubble 1929) ve Basamak 4-5 yazılmamıştır.

Uygulamada iki sadeleştirme vardır: model uzaklığı [Brout2022] denklemindeki gibi tek kırmızıya kaymayla (`zHD`) hesaplanır, güneş merkezli kırmızıya kayma çarpanı uygulanmaz; alt kırmızıya kayma eşiği 0,01 alınmıştır, makale bunu "~0.01" diye verir.

## Basamak 0: Hubble 1929

**Amaç.** Hız-uzaklık bağıntısını özgün veriyle kurmak; doğrusal uyum, belirsizlik ve sistematik hata kavramlarını en küçük veri setinde göstermek.

- Veri: Hubble'ın makalesindeki uzaklık ve dikine hız tablosu [Hubble1929]. Tablo makaleden elle aktarılır; aktarım ikinci bir okumayla denetlenir ve `veri/referans/` altına kaynağıyla kaydedilir.
- Yöntem: orijinden geçen ve serbest kesişimli doğrusal uyum; eğimin belirsizliği (analitik ve bootstrap).
- Tartışma: bulunan eğimin bugünkü değerin birkaç katı çıkmasının nedeni uzaklık ölçeğindeki sistematik hatadır. İstatistik belirsizlik küçük, sistematik hata büyüktür; bu ayrım projenin geri kalanının ana temasıdır.

**Kabul.** Makalede bildirilen eğim yeniden elde edilir; değer makaleden okunup teste yazılır.

Gereken modüller: `istatistik/uyum.py`, `io/tablo.py`.

## Basamak 1: Uzaklık hesaplayıcı

**Amaç.** `kozmoloji/friedmann.py` ve `kozmoloji/uzaklik.py`.

- `E(z)`, `D_C`, `D_M`, `D_A`, `D_L`, uzaklık modülü, geriye bakış zamanı, yaş [Hogg1999].
- Modeller: düz ΛCDM, eğrilikli ΛCDM, `wCDM`, `w0waCDM` [ChevallierPolarski2001; Linder2003].
- Vektörleştirilmiş integral: 1701 kırmızıya kayma için tek geçiş; olabilirlik binlerce kez çağrılacağı için hız ölçülür.

**Kabul.** `astropy.cosmology` ile `1e-8` bağıl uyum; Einstein-de Sitter kapalı çözümleri; Hogg'un şekillerinin yeniden üretimi.

## Basamak 2: Pantheon+ ile düz ΛCDM

**Amaç.** Süpernova Hubble diyagramından `Ωm` kestirimi.

- Veri: `Pantheon+SH0ES.dat` ve `Pantheon+SH0ES_STAT+SYS.cov` [Scolnic2022; Brout2022]. `arsiv/pantheon.py` dosyaları indirir, özetlerini kaydeder, tabloyu ve 1701 × 1701 kovaryansı okur.
- Seçim: yalnızca Hubble diyagramı analizinde düşük `z` kesimi (özel hızların baskın olduğu bölge dışarıda bırakılır). Eşik makaleden okunur [Brout2022].
- Model: `μ_model(z) = 5 log10(D_L(z; Ωm, H0)/10 pc)`. Işınım gücü uzaklığında güneş merkezli ve CMB çerçeveli kırmızıya kaymaların birlikte kullanımı: `D_L = (1 + z_hel) D_M(z_HD)` [Carr2022].
- Olabilirlik: `χ² = Δᵀ C⁻¹ Δ`; Cholesky çarpanı bir kez hesaplanır. `M_B` ile `H0` dejeneredir: bu basamakta birleşik parametre analitik olarak marjinalleştirilir [Conley2011].
- Çıkarım: önce ızgara taraması ve en büyük olabilirlik, sonra kendi topluluk örnekleyicimizle MCMC [GoodmanWeare2010]; `emcee` ile karşılaştırma [ForemanMackey2013].
- Tanılar: `R̂`, etkin örnek sayısı, zincir izleri [Vehtari2021]; artıkların kırmızıya kaymaya karşı grafiği; `χ²`/serbestlik derecesi [Andrae2010].

**Kabul.** Kestirilen `Ωm` ve belirsizliği yayımlanan düz ΛCDM sonucuyla belirsizlik içinde uyuşur; yayımlanan değer makaleden okunup teste yazılır [Brout2022]. Yalnızca istatistik kovaryansla yapılan uyumun belirsizliği tam kovaryanslı uyumunkinden küçük çıkar; fark sistematiklerin payını gösterir.

## Basamak 3: Sefeid çapasıyla H0

**Amaç.** Dejenerasyonu kırmak ve `H0` kestirmek.

- Kalibratör süpernovalar (`IS_CALIBRATOR = 1`, 77 satır) için model değeri yerine Sefeid konak uzaklık modülü (`CEPH_DIST`) kullanılır; artık `Δ_i = m_b_corr,i − M_B − μ_Sefeid,i` olur. Hubble akışı süpernovalarında `Δ_i = m_b_corr,i − M_B − μ_model(z_i)` [Brout2022]. Tanım, veri deposunun `README` dosyasından ve makaleden teyit edilecektir.
- Serbest parametreler: `H0`, `Ωm`, `M_B`.
- Uzantılar: `w0CDM`, `w0waCDM`; eğrilik.

**Kabul.** Yayımlanan `H0` değeri yeniden elde edilir; taramanın bildirdiği değerler `73,04 ± 1,04` [Riess2022] ve Pantheon+ ile birlikte `73,5 ± 1,1` (`w0CDM`) [Brout2022]. Kabul değeri kodlama sırasında makalelerden okunur.

## Basamak 4: Sistematikler ve seçim etkileri

Yeniden üretim tuttuktan sonra:

- Düşük `z` kesiminin, özel hız düzeltmesinin ve konak kütle basamağının sonuca etkisi.
- Sentetik Hubble diyagramı: bilinen kozmolojiyle üretilen veride akı sınırı uygulanır; Malmquist yanlılığının `H0` ve `w` üzerindeki etkisi ölçülür [KesslerScolnic2017].
- Derleme bağımlılığı: ikinci bir süpernova derlemesiyle yineleme.
- Kozmografik uyum: `q0` ve `j0` ile modelden bağımsız `H0` [Riess2022].

## Basamak 5: Birleşik sondalar (ileri aşama)

- BAO uzaklık ölçümlerinin eklenmesi; ses ufku `r_d` ile `H0` dejenerasyonu ve ters uzaklık merdiveni [DESI2025; DESI2025DR2; EisensteinHu1998].
- CMB sıkıştırılmış uzaklık önselleri [Planck2020].
- Kozmik kronometrelerle `H(z)` [Moresco2022].
- Yerel ve erken evren değerlerinin tek şekilde karşılaştırılması; gerilimin `σ` cinsinden ifadesi ve bu ifadenin varsayımları [DiValentino2021].

## Kendi gözlemimizle bağlantı

Yer tabanlı gözlem hattı tamamlandığında aynı proje ham veriyle beslenebilir:

- Bir Sefeidin ışık eğrisinin gözlenmesi, döneminin bulunması ve Leavitt yasasıyla uzaklığının kestirilmesi [LeavittPickering1912; Madore1982].
- Yakın bir Tür Ia süpernovanın çok bantlı ışık eğrisi, `sncosmo` ile uydurma ve uzaklık modülü [Guy2007].
- Gökada tayflarından kırmızıya kayma ölçümü ve SDSS değerleriyle karşılaştırma [TonryDavis1979].

Bu ölçümler `H0`'ı iyileştirmez; ölçüm zincirinin her halkasının anlaşıldığını gösterir.

## Çıktılar

- Hubble diyagramı ve artık grafiği; `Ωm-H0` ve `Ωm-w` sonsal dağılımları; yayımlanmış sonuçla karşılaştırma tablosu.
- Çalıştırma kaydı: veri özetleri, parametreler, zincirler, tanılar.
- Analizin dayandığı yayınların listesi (`kozmos.bibliography_raporu()`).

## Riskler

- Veri tanımlarının yanlış okunması (hangi kırmızıya kayma, hangi kadir sütunu, kalibratör kuralı) en olası hata kaynağıdır; önlem `README` ve makalenin okunması ve Basamak 2'nin yayımlanan değerle karşılaştırılmasıdır.
- 1701 boyutlu kovaryansın her adımda çözülmesi yavaştır; Cholesky çarpanı önceden hesaplanır.
- Yayımlanan analizdeki ek adımlar (yanlılık düzeltmeleri) hazır sütunlara gömülüdür; ışık eğrisi düzeyinden yeniden üretim bu projenin kapsamı dışındadır.
