using Emgu.CV;
using Emgu.CV.Structure;
using static Emgu.CV.CvInvoke;

Console.Write("Filename: ");
var filename = Console.ReadLine();
var input = Imread(filename);

// выравнивание гистограммы
var gray = new Mat();
var outEqHist = new Mat();
CvtColor(input, gray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
Imwrite(filename + ".gray.jpg", gray);
EqualizeHist(gray, outEqHist);
Imwrite(filename + ".outEqHist.jpg", outEqHist);

// локальное выравнивание над L каналом
var outLocalEqHist = new Mat();
CvtColor(input, outLocalEqHist, Emgu.CV.CvEnum.ColorConversion.Bgr2Lab);
var imageObject = outLocalEqHist.ToImage<Lab, byte>();
var channel = imageObject[0];
EqualizeHist(channel, channel);
imageObject[0] = channel;
CvtColor(imageObject, outLocalEqHist, Emgu.CV.CvEnum.ColorConversion.Lab2Bgr);
Imwrite(filename + ".outLocalEqHist.jpg", outLocalEqHist);

// гауссовское размытие
var outGaussBlur = new Mat();
GaussianBlur(input, outGaussBlur, new System.Drawing.Size(13, 13), 0);
Imwrite(filename + ".outGaussBlur.jpg", outGaussBlur);

// фильтр Собеля
GaussianBlur(input, outGaussBlur, new System.Drawing.Size(3, 3), 0); // для устранения шумов
CvtColor(outGaussBlur, gray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
var gradX = new Mat();
var gradY = new Mat();
var absGradX = new Mat();
var absGradY = new Mat();
Sobel(gray, gradX, Emgu.CV.CvEnum.DepthType.Cv16S, 1, 0);
Sobel(gray, gradY, Emgu.CV.CvEnum.DepthType.Cv16S, 0, 1);
ConvertScaleAbs(gradX, absGradX, 1, 0);
ConvertScaleAbs(gradY, absGradY, 1, 0);
var outSobel = new Mat();
AddWeighted(absGradX, 0.5, absGradY, 0.5, 0, outSobel);
Imwrite(filename + ".outSobel.jpg", outSobel);

// фильтр Лапласа
var outLap = new Mat();
var tempLap = new Mat();
Laplacian(gray, tempLap, Emgu.CV.CvEnum.DepthType.Cv16S);
ConvertScaleAbs(tempLap, outLap, 1, 0);
Imwrite(filename + ".outLap.jpg", outLap);
var moreLightPlease = outLap.ToImage<Gray, byte>();
moreLightPlease[0] *= 3; // увеличим НЕМНОГО яркость
Imwrite(filename + ".outLapMuchBetter.jpg", moreLightPlease);

// ничего не упало
Console.WriteLine("Successful");
