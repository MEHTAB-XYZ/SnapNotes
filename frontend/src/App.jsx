import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { jsPDF } from 'jspdf'
import notesIllustration from './assets/notes-illustration.png'
import './App.css'
import './notes.css'

function App() {
  const [files, setFiles] = useState([])
  const [topic, setTopic] = useState('')
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [fileError, setFileError] = useState('')
  const [previewUrls, setPreviewUrls] = useState([])
  const notesRef = useRef(null)

  // Create preview URLs for images
  const createPreviewUrls = (fileList) => {
    const newPreviewUrls = [];
    for (let i = 0; i < fileList.length; i++) {
      newPreviewUrls.push(URL.createObjectURL(fileList[i]));
    }
    return newPreviewUrls;
  }

  // Clean up preview URLs when component unmounts
  useEffect(() => {
    return () => {
      previewUrls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [previewUrls]);

  // Handle paste events for clipboard images
  useEffect(() => {
    const handlePaste = (e) => {
      if (e.clipboardData && e.clipboardData.items) {
        const items = e.clipboardData.items;
        const newFiles = [...files];
        let hasAddedFiles = false;

        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf('image') !== -1) {
            const file = items[i].getAsFile();
            // Check file size (5MB)
            if (file.size > 5 * 1024 * 1024) {
              setFileError('File size exceeds 5MB limit');
              continue;
            }
            newFiles.push(file);
            hasAddedFiles = true;
          }
        }

        if (hasAddedFiles) {
          setFiles(newFiles);
          setPreviewUrls(createPreviewUrls(newFiles));
          setFileError('');
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [files]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFileError('');
    
    if (selectedFiles.length === 0) {
      return;
    }
    
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const validFiles = [];
    
    // Validate each file
    for (const file of selectedFiles) {
      // Check file type
      if (!validTypes.includes(file.type)) {
        setFileError('Please upload valid image files (JPG, JPEG, PNG)');
        continue;
      }
      
      // Check file size (5MB)
      if (file.size > 5 * 1024 * 1024) {
        setFileError('One or more files exceed 5MB limit');
        continue;
      }
      
      validFiles.push(file);
    }
    
    if (validFiles.length > 0) {
      const newFiles = [...files, ...validFiles];
      setFiles(newFiles);
      setPreviewUrls(createPreviewUrls(newFiles));
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (files.length === 0) {
      setFileError('Please upload at least one image file')
      return
    }
    
    if (!topic.trim()) {
      setError('Please enter a subject/topic')
      return
    }
    
    setLoading(true)
    setError('')
    setNotes('')
    
    // Process each file and combine the results
    try {
      let combinedNotes = '';
      
      for (let i = 0; i < files.length; i++) {
        const formData = new FormData()
        formData.append('file', files[i])
        formData.append('topic', topic)
        
        const response = await axios.post('http://localhost:8000/generate-notes', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        // Clean up code block markers and combine notes from multiple images
        const cleanedNotes = response.data.notes
          .replace(/```[a-z]*\n/g, '') // Remove opening code block markers with language
          .replace(/```\n/g, '') // Remove opening code block markers without language
          .replace(/```/g, '') // Remove closing code block markers
          .trim()
        
        combinedNotes += cleanedNotes + '\n';
      }
      
      setNotes(combinedNotes)
    } catch (err) {
      console.error('Error generating notes:', err)
      setError(err.response?.data?.error || 'Failed to generate notes. Please try again.')
    } finally {
      setLoading(false)
    }
  }
  
  const removeFile = (index) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
    
    const newPreviewUrls = [...previewUrls];
    URL.revokeObjectURL(newPreviewUrls[index]);
    newPreviewUrls.splice(index, 1);
    setPreviewUrls(newPreviewUrls);
  }

  const parseNotesContent = (htmlContent) => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');
    const elements = doc.body.children;
    const parsedContent = [];

    const parseElement = (element) => {
      const type = element.tagName.toLowerCase();
      const text = element.textContent.trim();

      switch (type) {
        case 'h1':
          return { type: 'heading1', text };
        case 'h2':
          return { type: 'heading2', text };
        case 'h3':
          return { type: 'heading3', text };
        case 'p':
          return { type: 'paragraph', text };
        case 'ul':
          return {
            type: 'bulletList',
            items: Array.from(element.children).map(li => li.textContent.trim())
          };
        case 'ol':
          return {
            type: 'numberList',
            items: Array.from(element.children).map(li => li.textContent.trim())
          };
        default:
          return { type: 'paragraph', text };
      }
    };

    Array.from(elements).forEach(element => {
      parsedContent.push(parseElement(element));
    });

    return parsedContent;
  };

  const generatePDF = async () => {
    try {
      // Create new PDF document
      const pdf = new jsPDF({
        unit: 'pt',
        format: 'a4',
        orientation: 'portrait'
      });

      // Set document properties
      pdf.setProperties({
        title: `${topic} Notes`,
        subject: 'Generated Notes',
        author: 'SnapNotes by MoonLit',
        keywords: 'notes, study',
        creator: 'SnapNotes'
      });

      // Initialize position tracking
      let yPos = 40;
      const margin = 40;
      const pageHeight = pdf.internal.pageSize.height;
      const pageWidth = pdf.internal.pageSize.width;
      const contentWidth = pageWidth - (margin * 2);

      // Add title at the top of first page
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(16);
      pdf.setTextColor(0, 0, 0);
      pdf.text(`${topic} Notes`, margin, yPos);
      yPos += 25;

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      pdf.text(`Generated on ${new Date().toLocaleDateString()}`, margin, yPos);
      yPos += 30;

      // Parse HTML content
      const parsedContent = parseNotesContent(notes);

      // Process each content block
      parsedContent.forEach(content => {
        // Check if we need a new page
        if (yPos > pageHeight - 60) {
          pdf.addPage();
          yPos = 40;
        }

        switch (content.type) {
          case 'heading1':
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(14);
            pdf.setTextColor(0, 0, 0);
            
            const h1Lines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(h1Lines, margin, yPos);
            yPos += (h1Lines.length * 20) + 10;
            break;

          case 'heading2':
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(12);
            pdf.setTextColor(0, 0, 0);
            
            const h2Lines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(h2Lines, margin, yPos);
            yPos += (h2Lines.length * 18) + 8;
            break;

          case 'heading3':
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            const h3Lines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(h3Lines, margin, yPos);
            yPos += (h3Lines.length * 16) + 6;
            break;

          case 'paragraph':
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            const pLines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(pLines, margin, yPos);
            yPos += (pLines.length * 14) + 10;
            break;

          case 'bulletList':
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            content.items.forEach((item, index) => {
              if (yPos > pageHeight - 60) {
                pdf.addPage();
                yPos = 40;
              }
              
              const bulletPoint = '•';
              const itemLines = pdf.splitTextToSize(item, contentWidth - 15);
              
              pdf.text(bulletPoint, margin, yPos);
              pdf.text(itemLines, margin + 15, yPos);
              yPos += (itemLines.length * 14) + 5;
            });
            yPos += 5;
            break;

          case 'numberList':
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            content.items.forEach((item, index) => {
              if (yPos > pageHeight - 60) {
                pdf.addPage();
                yPos = 40;
              }
              
              const number = `${index + 1}.`;
              const itemLines = pdf.splitTextToSize(item, contentWidth - 20);
              
              pdf.text(number, margin, yPos);
              pdf.text(itemLines, margin + 20, yPos);
              yPos += (itemLines.length * 14) + 5;
            });
            yPos += 5;
            break;
        }
      });

      // Open in new tab as blob URL
      const pdfBlob = new Blob([pdf.output('blob')], { type: 'application/pdf' });
      const blobUrl = URL.createObjectURL(pdfBlob);
      window.open(blobUrl, '_blank');
    } catch (error) {
      console.error('Error generating PDF:', error);
      setError('Failed to generate PDF. Please try again.');
    }
  };

  const handleDownload = async () => {
    try {
      // Create new PDF document
      const pdf = new jsPDF({
        unit: 'pt',
        format: 'a4',
        orientation: 'portrait'
      });

      // Set document properties
      pdf.setProperties({
        title: `${topic} Notes`,
        subject: 'Generated Notes',
        author: 'SnapNotes by MoonLit',
        keywords: 'notes, study',
        creator: 'SnapNotes'
      });

      // Initialize position tracking
      let yPos = 40;
      const margin = 40;
      const pageHeight = pdf.internal.pageSize.height;
      const pageWidth = pdf.internal.pageSize.width;
      const contentWidth = pageWidth - (margin * 2);

      // Add title at the top of first page
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(16);
      pdf.setTextColor(0, 0, 0);
      pdf.text(`${topic} Notes`, margin, yPos);
      yPos += 25;

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      pdf.text(`Generated on ${new Date().toLocaleDateString()}`, margin, yPos);
      yPos += 30;

      // Parse HTML content
      const parsedContent = parseNotesContent(notes);

      // Process each content block
      parsedContent.forEach(content => {
        // Check if we need a new page
        if (yPos > pageHeight - 60) {
          pdf.addPage();
          yPos = 40;
        }

        switch (content.type) {
          case 'heading1':
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(14);
            pdf.setTextColor(0, 0, 0);
            
            const h1Lines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(h1Lines, margin, yPos);
            yPos += (h1Lines.length * 20) + 10;
            break;

          case 'heading2':
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(12);
            pdf.setTextColor(0, 0, 0);
            
            const h2Lines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(h2Lines, margin, yPos);
            yPos += (h2Lines.length * 18) + 8;
            break;

          case 'heading3':
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            const h3Lines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(h3Lines, margin, yPos);
            yPos += (h3Lines.length * 16) + 6;
            break;

          case 'paragraph':
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            const pLines = pdf.splitTextToSize(content.text, contentWidth);
            pdf.text(pLines, margin, yPos);
            yPos += (pLines.length * 14) + 10;
            break;

          case 'bulletList':
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            content.items.forEach((item, index) => {
              if (yPos > pageHeight - 60) {
                pdf.addPage();
                yPos = 40;
              }
              
              const bulletPoint = '•';
              const itemLines = pdf.splitTextToSize(item, contentWidth - 15);
              
              pdf.text(bulletPoint, margin, yPos);
              pdf.text(itemLines, margin + 15, yPos);
              yPos += (itemLines.length * 14) + 5;
            });
            yPos += 5;
            break;

          case 'numberList':
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(11);
            pdf.setTextColor(0, 0, 0);
            
            content.items.forEach((item, index) => {
              if (yPos > pageHeight - 60) {
                pdf.addPage();
                yPos = 40;
              }
              
              const number = `${index + 1}.`;
              const itemLines = pdf.splitTextToSize(item, contentWidth - 20);
              
              pdf.text(number, margin, yPos);
              pdf.text(itemLines, margin + 20, yPos);
              yPos += (itemLines.length * 14) + 5;
            });
            yPos += 5;
            break;
        }
      });

      // Download the PDF
      pdf.save(`${topic.toLowerCase().replace(/\s+/g, '-')}-notes.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
      setError('Failed to generate PDF. Please try again.');
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-gray-50 flex">
      {/* Left Panel */}
      <div className="w-1/2 h-full p-6 flex flex-col">
        <div className="flex-none mb-4">
          <h1 className="text-3xl font-extrabold text-gray-900">
            SnapNotes
            <span className="text-lg font-medium text-gray-500 ml-2">by MoonLit</span>
          </h1>
        </div>
        
        <form onSubmit={handleSubmit} className="flex-1 flex flex-col min-h-0 overflow-auto">
          {/* Main content area with fixed heights and scrolling */}
          <div className="flex-1 flex flex-col space-y-4">
            {/* Illustration with fixed height */}
            <div className="h-56 flex-none bg-white rounded-xl p-4 flex items-center justify-center shadow-sm">
              <img
                src={notesIllustration}
                alt="Notes illustration"
                className="h-48 w-auto object-contain"
              />
            </div>

            {/* Upload area with scrollable image preview */}
            <div className="flex-1 flex flex-col space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-xl bg-white p-4 transition-all duration-300 hover:border-indigo-300">
                <div className="flex flex-col items-center space-y-4">
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 px-4 py-2 border border-gray-300 hover:border-indigo-300 hover:shadow-sm transition-all duration-300"
                  >
                    <span>Upload files</span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      accept="image/jpeg,image/jpg,image/png"
                      onChange={handleFileChange}
                      multiple
                    />
                  </label>
                  <p className="text-sm text-gray-500">JPG, JPEG, PNG up to 5MB each</p>
                  <p className="text-sm text-blue-500">You can also paste images with Ctrl+V</p>
                </div>
              </div>

              {/* Scrollable image preview area */}
              <div className="flex-1 overflow-y-auto p-4 border border-gray-200 rounded-xl bg-gray-50">
                {previewUrls.length > 0 ? (
                  <div className="grid grid-cols-3 gap-4">
                    {previewUrls.map((url, index) => (
                      <div key={index} className="relative group">
                        <div className="aspect-[4/3] rounded-lg overflow-hidden border border-gray-200">
                          <img 
                            src={url} 
                            alt={`Preview ${index}`} 
                            className="w-full h-full object-cover" 
                          />
                        </div>
                        <button
                          type="button"
                          onClick={() => removeFile(index)}
                          className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-sm opacity-0 group-hover:opacity-100 transition-all duration-300 hover:bg-red-600"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 text-center">No images uploaded yet.</p>
                )}
              </div>
            </div>

            {/* Topic input and submit */}
            <div className="flex-none space-y-4">
              <div>
                <label htmlFor="topic" className="block text-sm font-medium text-gray-700 mb-1">
                  Subject/Topic
                </label>
                <input
                  type="text"
                  name="topic"
                  id="topic"
                  className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full text-sm border-gray-300 rounded-lg p-3 border transition-all duration-300 hover:shadow-md"
                  placeholder="e.g. Thermodynamics, WW2 History"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
              </div>
              
              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center px-6 py-2.5 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 hover:shadow-md transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </>
                  ) : 'Generate Notes'}
                </button>
              </div>
              
              {error && (
                <div className="text-sm text-red-600 text-center bg-red-50 rounded-lg p-2">
                  {error}
                </div>
              )}
            </div>
          </div>
        </form>
      </div>

      {/* Right Panel */}
      <div className="w-1/2 h-full bg-white p-6 border-l border-gray-200 overflow-hidden flex flex-col">
        {notes ? (
          <>
            <div className="flex-none flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-900">Generated Notes</h2>
              <div className="flex space-x-2">
                <button
                  onClick={generatePDF}
                  className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-md text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors duration-200"
                >
                  <svg className="-ml-0.5 mr-1.5 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 0115 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  Preview PDF
                </button>
                <button
                  onClick={handleDownload}
                  className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 transition-colors duration-200"
                >
                  <svg className="-ml-0.5 mr-1.5 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download PDF
                </button>
              </div>
            </div>
            
            <div 
              ref={notesRef} 
              className="flex-1 overflow-y-auto bg-white rounded-xl prose prose-indigo max-w-none notes-container"
              dangerouslySetInnerHTML={{ __html: notes }}
            />
          </>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <svg className="h-24 w-24 mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-lg">Your generated notes will appear here</p>
            <p className="text-sm mt-2 text-gray-400">Upload images and click Generate to start</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App