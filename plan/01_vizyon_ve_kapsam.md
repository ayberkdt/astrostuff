# Vizyon ve kapsam

## Amaç

Gökbilim ve astrofizik hesaplarını, gözlem verisinin indirgenmesinden kozmolojik parametre kestirimine kadar, kendi yazdığımız bir Python kitaplığında toplamak. Kitaplığın ayırt edici özelliği izlenebilirliktir: her fonksiyon uyguladığı denklemin yayımlanmış kaynağını denklem numarasıyla gösterir ve bağımsız bir uygulamaya karşı sınanır.

Olgun kitaplıklar bu hesapların çoğunu yapmaktadır ([02_saha_arastirmasi/01_yazilim_ekosistemi.md](02_saha_arastirmasi/01_yazilim_ekosistemi.md)). Yeniden yazmanın gerekçesi üçtür:

1. **Anlama.** Bir indirgeme zincirinin ya da uzaklık hesabının her adımını yazan kişi, sonucun hangi varsayıma ne kadar duyarlı olduğunu bilir.
2. **İzlenebilirlik.** Hazır kitaplıkta bir sayının hangi denklemden, hangi sabitle, hangi yaklaşımla çıktığını bulmak kaynak kodu kazmayı gerektirir. Burada bu bilgi fonksiyonun arayüzündedir.
3. **Doğrulanmış yeniden üretim.** Olgun kitaplıklar test kâhini olarak kullanılır; iki bağımsız uygulamanın uyuşması tek birinin sonucundan güçlü kanıttır.

## Kapsam

Dört katman:

- **Hesap altyapısı:** sabitler, birimler, açılar, zaman ölçekleri, koordinat dönüşümleri, sayısal yöntemler, istatistik.
- **Gözlem:** planlama, dedektör ve atmosfer modeli, görüntü indirgeme, fotometri, astrometri, tayf, zaman serisi. "Yer gözlemi" yer tabanlı (teleskopla, yeryüzünden) astronomik gözlem olarak yorumlanmıştır; Dünya'nın uydudan gözlemi (uzaktan algılama) kapsamda değildir.
- **Astrofizik modelleme:** ışınım, yörünge mekaniği, yıldız yapısı ve popülasyonları, gökada dinamiği, kozmoloji.
- **Uçtan uca analiz:** açık veriden yayımlanmış sonuçların yeniden üretimi; ilk hedef Hubble diyagramı ve Hubble sabiti.

## Kapsam dışı

- Büyük kodların yeniden yazımı: yıldız evrimi (MESA), Boltzmann çözücüler (CAMB, CLASS), hidrodinamik ve kozmolojik benzetim, ışınım aktarımı kodları, süpernova ışık eğrisi modelleri (SALT). Bunlar kullanılır ya da çıktıları okunur.
- Genel makine öğrenmesi algoritmaları; `scikit-learn` kullanılır.
- Teleskop ve kamera denetimi, gerçek zamanlı işletim.
- Grafik arayüz.

Sınırın ölçütü: bir algoritma birkaç yüz satırda, yayımlanmış denklemlerden yazılabiliyor ve bağımsız bir uygulamaya karşı sınanabiliyorsa kapsamdadır.

## Başarı ölçütleri

- Her açık fonksiyonun kaynağı, geçerlilik aralığı ve doğrulama testi vardır; bu kural makineyle denetlenir.
- Her modül bir kâhine karşı belgelenmiş tolerans içinde uyuşur.
- Her aşama yayımlanmış bir sonucu yeniden üreten bir projeyle kapanır.
- Bir analiz, çalıştırma kaydından yeniden koşturulduğunda aynı sonucu verir.

## Hedef kullanıcı

Kitaplığı yazan ve kullanan kişi aynıdır: gözlem yapan, veriyi kendi indirgeyen ve sonucu yayımlanmış değerlerle karşılaştırmak isteyen araştırmacı. Kod öğretici olacak kadar okunur, bilim sonucu verecek kadar doğru olmalıdır; ikisi çatıştığında doğruluk önce gelir ve okunurluk belge dizgisiyle sağlanır.

## Açık kararlar

- **Paket adı:** `kozmos` (PyPI'da boş).
- **Tanımlayıcı dili:** Türkçe ASCII ya da İngilizce. Belgelerdeki örnekler Türkçedir.
- **Lisans:** kod yayımlanacaksa seçilmelidir; kâhin kütüphanelerinin çoğu BSD lisanslıdır. SOFA kendi lisansıyla dağıtılır; koşulları bu çalışmada okunmamıştır ve SOFA kaynak kodundan aktarım yapılmadan önce okunmalıdır. Algoritmalar yayımlanmış makale ve kitaplardan yazılırsa bu sorun doğmaz.
- **Hızlandırma:** NumPy yeterli olmadığında Numba ya da JAX; karar ölçümle verilir.
