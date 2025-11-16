import { Upload, X } from 'lucide-react';
import { useState, useRef } from 'react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  isAnalyzing: boolean;
}

export default function ImageUpload({ onImageSelect, selectedImage, isAnalyzing }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const clearImage = () => {
    setPreview(null);
    onImageSelect(null as any);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      {!preview ? (
        <div
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="border-3 border-dashed border-blue-300 rounded-2xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all duration-300 bg-gradient-to-br from-blue-50 to-cyan-50"
        >
          <Upload className="mx-auto h-16 w-16 text-blue-500 mb-4" />
          <p className="text-xl font-semibold text-gray-700 mb-2">
            Upload Brain MRI Scan
          </p>
          <p className="text-sm text-gray-500 mb-4">
            Drag and drop or click to browse
          </p>
          <p className="text-xs text-gray-400">
            Supported formats: JPG, PNG, DICOM
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            disabled={isAnalyzing}
          />
        </div>
      ) : (
        <div className="relative rounded-2xl overflow-hidden shadow-2xl border-4 border-blue-200">
          <img
            src={preview}
            alt="MRI Preview"
            className="w-full h-96 object-contain bg-black"
          />
          {!isAnalyzing && (
            <button
              onClick={clearImage}
              className="absolute top-4 right-4 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg transition-all duration-200 hover:scale-110"
            >
              <X className="h-5 w-5" />
            </button>
          )}
          {isAnalyzing && (
            <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-400 mx-auto mb-4"></div>
                <p className="text-white text-lg font-semibold">Analyzing MRI...</p>
                <p className="text-blue-300 text-sm mt-2">Running AI models</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
