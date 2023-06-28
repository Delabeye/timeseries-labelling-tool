
% ddict = pyrunfile("_tmp.py", "ddict")
ddict, llist = pyrunfile("_tmp.py", "z", x=1, y=2)


% rootdir = strsplit(fileparts(mfilename('fullpath')), '/');
% rootdir = strjoin(rootdir(1:end-1), "/");
% 
% % data = [
% %     parquetread(strjoin({rootdir; '/sample_dataset/data/trial42/part.0.parquet'}, "/"));
% %     parquetread(strjoin({rootdir; '/sample_dataset/data/trial42/part.1.parquet'}, "/"));
% % ];
% 
% metadata = parquetmeta(strjoin({rootdir; '/sample_dataset/data/trial42/_metadata'}, "/"))
% 
% 
% function metadata = parquetmeta(filename)
%     fid = fopen(filename);
%     fseek(fid, -8, 'eof');
%     footer_bytes = fread(fid, 1, 'uint32', 'ieee-le');
%     fseek(fid, -footer_bytes, 'eof');
%     footer = fread(fid, [1, footer_bytes], '*char');
%     fclose(fid);
%     start_idx = find(footer == '{', 1, 'first');
%     end_idx = find(footer == '}', 1, 'last');
%     metadata = struct;
%     
% 
%     if ~isempty(start_idx) && ~isempty(end_idx)
%         json_str = footer(start_idx: end_idx)
%         jsondecode(json_str)
%         try %#ok<TRYNC>
%             metadata = jsondecode(footer(start_idx: end_idx));
%         end
%     end
% end


